from net.loggers import *
from net.batcher import Batcher


class Trainer:

    def __init__(self, model, train_data, val_data, epochs, batch_size, callbacks):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.epochs_amount = epochs
        self.batch_size = batch_size
        self.logger = LossAccLogger()

        self.callbacks = callbacks
        for cb in self.callbacks:
            cb.set_trainer(self)

    def update_epoch_log(self, tag, loss, accuracy):
        self.logger.add_loss_per_epoch(tag, loss)
        self.logger.add_accuracy_per_epoch(tag, accuracy)

    def update_batch_log(self, loss):
        self.logger.add_loss_per_batch(loss)

    def validate(self):
        x_val, y_val = self.val_data
        out, _ = self.model.forward_pass(x_val, learning_phase=False)
        loss_val, _ = self.model.compute_loss(out, y_val)

        y_pred = self.model.predict_classes(x_val)
        metric_measures = self.model.eval_metrics(y_pred, y_val)

        return metric_measures, loss_val

    def train_loop(self):
        x_train, y_train = self.train_data
        b = Batcher(x_train, y_train)
        self._callback('on_train_begin')

        for epoch in range(self.epochs_amount):
            self._callback('on_epoch_begin', epoch=epoch + 1)

            self._callback('on_train_epoch_begin', epoch=epoch + 1)
            while b.epoch() < epoch + 1:
                x, y = b(self.batch_size)
                self._callback('on_batch_begin', batch=(x, y))
                train_loss = self.model.train(x, y)
                train_pred = self.model.predict_classes(x)
                self._callback('on_batch_end', batch=(x, y), epoch=epoch + 1,
                               train_pred=train_pred, train_loss=train_loss)
            self._callback('on_train_epoch_end', epoch=epoch + 1)

            self._callback('on_epoch_end', epoch=epoch + 1)
            continue

        self._callback('on_train_end')

    def _callback(self, func, *args, **kwargs):
        kwargs['trainer'] = self
        ls = []
        for cb in self.callbacks:
            result = getattr(cb, func)(*args, **kwargs)
            if result is not None:
                ls.append(result)
        return ls
