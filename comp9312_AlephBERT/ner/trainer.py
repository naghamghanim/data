import os
import torch
import logging
import numpy as np
from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from seqeval.scheme import IOB2

logger = logging.getLogger(__name__)


class BertTrainer:
    def __init__(
        self,
        model=None,
        max_epochs=50,
        optimizer=None,
        loss=None,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        output_path=None,
        clip=5,
        patience=5,
    ):
        self.model = model
        self.max_epochs = max_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.output_path = output_path
        self.clip = clip
        self.patience = patience
        self.timestep = 0
        self.epoch = 0

    def save(self):
        """
        Save model checkpoint
        :return:
        """
        filename = os.path.join(self.output_path, "model.pt")

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        logger.info("Saving checkpoint to %s", filename)
        torch.save(checkpoint, filename)

    def compute_metrics(self, segments):
        """
        Compute macro and micro metrics
        :param y_true: List - ground truth labels
        :param y_pred: List - prediucted labels
        :return:
        """
        y_true = [[token.gold_tag for token in segment] for segment in segments]
        y_pred = [[token.pred_tag for token in segment] for segment in segments]

        logger.info("\n" + classification_report(y_true, y_pred))

        metrics = {
            "micro_f1": f1_score(y_true, y_pred, average="micro", scheme=IOB2),
            "macro_f1": f1_score(y_true, y_pred, average="macro", scheme=IOB2),
            "weights_f1": f1_score(y_true, y_pred, average="weighted", scheme=IOB2),
            "precision": precision_score(y_true, y_pred, scheme=IOB2),
            "recall": recall_score(y_true, y_pred, scheme=IOB2),
            "accuracy": accuracy_score(y_true, y_pred),
        }

        return metrics

    def segments_to_file(self, segments, filename):
        """
        Write segments to file
        :param segments: [List[arabiner.data.dataset.Token]] - list of list of tokens
        :param filename: str - output filename
        :return: None
        """
        with open(filename, "w") as fh:
            results = "\n\n".join(
                [
                    "\n".join([f"{t.text} {t.gold_tag} {t.pred_tag}" for t in segment])
                    for segment in segments
                ]
            )
            fh.write("Token\tGold Tag\tPredicted Tag\n")
            fh.write(results)
            logging.info("Predictions written to %s", filename)

    def train(self):
        best_val_loss, test_loss = np.inf, np.inf
        num_train_batch = len(self.train_dataloader)

        for epoch_index in range(self.max_epochs):
            self.epoch = epoch_index
            train_loss = 0

            for batch_index, batch in enumerate(self.train_dataloader, 1):
                _, labels, _, _, logits = self.classify(batch)
                self.timestep += 1
                batch_loss = self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
                batch_loss.backward()
                self.optimizer.step()
                train_loss += batch_loss.item()

                if self.timestep % 10 == 0:
                    logger.info(
                        "Epoch %d | Batch %d/%d | Timestep %d | Loss %f",
                        epoch_index,
                        batch_index,
                        num_train_batch,
                        self.timestep,
                        batch_loss.item(),
                    )

            train_loss /= num_train_batch

            logger.info("** Evaluating on validation dataset **")
            segments, val_loss = self.eval(self.val_dataloader)
            val_metrics = self.compute_metrics(segments)

            logger.info(
                "Epoch %d | Timestep %d | Train Loss %f | Val Loss %f | F1 Micro %f",
                epoch_index,
                self.timestep,
                train_loss,
                val_loss,
                val_metrics["micro_f1"],
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info("** Validation improved, evaluating test data **")
                segments, test_loss = self.eval(self.test_dataloader)
                self.segments_to_file(
                    segments, os.path.join(self.output_path, "predictions.txt")
                )
                test_metrics = self.compute_metrics(segments)

                logger.info(
                    f"Epoch %d | Timestep %d | Test Loss %f | F1 Micro %f",
                    epoch_index,
                    self.timestep,
                    test_loss,
                    test_metrics["micro_f1"],
                )

                self.save()
            else:
                self.patience -= 1

            # No improvements, terminating early
            if self.patience == 0:
                logger.info("Early termination triggered")
                break

    def classify(self, batch, is_train=True):
        """
        Given a dataloader containing segments, predict the tags
        :param dataloader: torch.utils.data.DataLoader
        :param is_train: boolean - True for training model, False for evaluation
        :return: Iterator
                    subwords (B x T x NUM_LABELS)- torch.Tensor - BERT subword ID
                    gold_tags (B x T x NUM_LABELS) - torch.Tensor - ground truth tags IDs
                    tokens - List[arabiner.data.dataset.Token] - list of tokens
                    valid_len (B x 1) - int - valiud length of each sequence
                    logits (B x T x NUM_LABELS) - logits for each token and each tag
        """
        subwords, labels, segments, valid_len = batch
        self.model.train(is_train)

        if torch.cuda.is_available():
            subwords = subwords.cuda()
            labels = labels.cuda()

        if is_train:
            self.optimizer.zero_grad()
            logits = self.model(subwords)
        else:
            with torch.no_grad():
                logits = self.model(subwords)

        return subwords, labels, segments, valid_len, logits

    def eval(self, dataloader):
        golds, preds, valid_lens, segments = list(), list(), list(), list()
        loss = 0

        for batch in dataloader:
            _, labels, batch_segments, valid_len, logits = self.classify(
                batch, is_train=False
            )
            batch_loss = self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1))

            loss += batch_loss
            valid_lens += valid_len
            preds += torch.argmax(logits, dim=2)
            segments += batch_segments

        loss /= len(dataloader)

        # Update segments, attach predicted tags to each token
        segments = self.to_segments(segments, preds, valid_lens, dataloader.dataset.transform.vocab)

        return segments, loss

    def to_segments(self, segments, preds, valid_lens, vocab):
        tag_itos = vocab.tags.get_itos()
        tagged_segments = list()

        for segment, pred, valid_len in zip(segments, preds, valid_lens):
            tagged_segment = list()
            pred = pred.detach().cpu().numpy().tolist()
            # First, the token at 0th index [CLS] and token at nth index [SEP]
            pred = pred[1:valid_len - 1]
            segment = segment[1:valid_len - 1]

            for token, pred_id in zip(segment, pred):
                if token.text != "<UNK>":
                    token.pred_tag = tag_itos[pred_id]
                    tagged_segment.append(token)

            tagged_segments.append(tagged_segment)

        return tagged_segments
