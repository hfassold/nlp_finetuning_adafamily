# Start : Import the packages
import pandas as pd
import os
import pathlib
import zipfile
import wget
import gdown
import torch
from torch import nn
from torch import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# FAH
# Import the function which is will replace the 'forward' function of class 'DistilBertForSequenceClassification' (from huggingface package)
# The technique is called 'monkey-patching', see "Tips&Tricks" in our WIKI page at https://digital-wiki.joanneum.at/pages/viewpage.action?pageId=98697788
import types
from transformers import DistilBertForSequenceClassification
from JrsDistilBertForSequenceClassification import forwardDistilBertForSequenceClassificationPatched
# 'AdaFamily' optimizer
from my_adafamily import AdaFamily
# ~FAH

# FAH: minibatch trimming
import mbtrim


global_random_seed = 42
#global_random_seed = 43
#global_random_seed = 44

global_do_minibatch_trimming = False
global_enable_adafamily = False

# is a global variable - see https://stackoverflow.com/questions/423379/using-global-variables-in-a-function
trainer = None

# End : Import the packages
# %%
# Remark: pl.LightningModule is derived from torch.nn.Module It has additional methods that are part of the
# lightning interface and that need to be defined by the user. Having these additional methods is very useful
# for several reasons:
# 1. Reasonable Expectations: Once you know the pytorch-lightning-system you more easily read other people's
#    code, because it is always structured in the same way
# 2. Less Boilerplate: The additional class methods make pl.LightningModule more powerful than the nn.Module
#    from plain pytorch. This means you have to write less of the repetitive boilerplate code
# 3. Perfact for the development lifecycle Pytorch Lightning makes it very easy to switch from cpu to gpu/tpu.
#    Further it supplies method to quickly run your code on a fraction of the data, which is very useful in
#    the development process, especially for debugging


class Model(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        # a very useful feature of pytorch lightning  which leads to the named variables that are passed in
        # being available as self.hparams.<variable_name> We use this when refering to eg
        # self.hparams.learning_rate

        # freeze
        self._frozen = False

        # eg https://github.com/stefan-it/turkish-bert/issues/5
        config = AutoConfig.from_pretrained(self.hparams.pretrained,
                                            num_labels=5,  # 1 implies regression
                                            output_attentions=False,
                                            output_hidden_states=False)


        print(config)

        A = AutoModelForSequenceClassification
        self.model = A.from_pretrained(self.hparams.pretrained, config=config)

        # FAH See my paper about minibatch trimming at https://arxiv.org/abs/2110.13058
        self.model.jrs_do_minibatch_trimming = global_do_minibatch_trimming

        # FAH Added new parameter by me (currently implemented ONLY for DistilBert model!)
        # If set false, the per-sample loss is returned (for each sample of the mini-batch),
        # so NO reduction (via 'mean' operation) is applied
        # This is needed by use for implementing 'mini-batch trimming)
        # Note only my monkey-patched (DistilBert) class supports this  parameter !!
        self.model.jrs_do_loss_reduction = True

        # FAH
        if self.model.jrs_do_minibatch_trimming:
            # if we have enable minibatch trimming, we HAVE to disable the reduction (over minibatch)
            self.model.jrs_do_loss_reduction = False
            print("\n\n*** Mini-batch trimming is _ENABLED_ *** \n\n")

        # Do a 'monkey-patching' of class 'DistilBertForSequenceClassification' (from huggingface 'transformers' package)
        # and replace its original 'forward' function with our special forward function 'forwardDistilBertForSequenceClassificationPatched'
        # We do monkey-patching because we can NOT derive from that class (because we want to use 'A.from_pretrained'
        # for construction of the model).
        # Taken from answer of 'taleinat' (not you have to import also the 'types' package) at
        # https://stackoverflow.com/questions/38485123/monkey-patching-bound-methods-in-python
        # See also 'Tips&Tricks' at our WIKI page at https://digital-wiki.joanneum.at/pages/viewpage.action?pageId=98697788
        if isinstance(self.model, DistilBertForSequenceClassification):
            self.model.forward = types.MethodType(forwardDistilBertForSequenceClassificationPatched, self.model)
            print ("Replacing the original <forward> method of <DistilBertForSequenceClassification> with our custom function ...")

        print('Model Type', type(self.model))

        # Possible choices for pretrained are:
        # distilbert-base-uncased
        # bert-base-uncased

        # The BERT paper says: "[The] pre-trained BERT model can be fine-tuned with just one additional output
        # layer to create state-of-the-art models for a wide range of tasks, such as question answering and
        # language inference, without substantial task-specific architecture modifications."
        #
        # Huggingface/transformers provides access to such pretrained model versions, some of which have been
        # published by various community members.
        #
        # BertForSequenceClassification is one of those pretrained models, which is loaded automatically by
        # AutoModelForSequenceClassification because it corresponds to the pretrained weights of
        # "bert-base-uncased".
        #
        # Huggingface says about BertForSequenceClassification: Bert Model transformer with a sequence
        # classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE
        # tasks."

        # This part is easy  we instantiate the pretrained model (checkpoint)

        # But it's also incredibly important, e.g. by using "bert-base-uncased, we determine, that that model
        # does not distinguish between lower and upper case. This might have a significant impact on model
        # performance!!!

    def forward(self, batch):
        # there are some choices, as to how you can define the input to the forward function I prefer it this
        # way, where the batch contains the input_ids, the input_put_mask and sometimes the labels (for
        # training)

        b_input_ids = batch[0]
        b_input_mask = batch[1]

        has_labels = len(batch) > 2

        b_labels = batch[2] if has_labels else None

        # FAH IMPORTANT: Unfortunately, it seems that we cannot add our 'custom loss' (minibatch-trimming)
        # when using the pytorch lightning optimizer pipeline. We would need to use the huggingface trainer,
        # as explained in https://discuss.huggingface.co/t/finetuning-bart-using-custom-loss/4060/4
        res = self.model(b_input_ids,
                                      attention_mask=b_input_mask,
                                      labels=b_labels)

        # there are labels in the batch, this indicates: training for the BertForSequenceClassification model:
        # it means that the model returns tuples, where the first element is the training loss and the second
        # element is the logits
        if has_labels:
            loss, logits = res['loss'], res['logits']

        # there are labels in the batch, this indicates: prediction for the BertForSequenceClassification
        # model: it means that the model returns simply the logits

        if not has_labels:
            loss, logits = None, res['logits']

        return loss, logits

    def training_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class stipulates you to overwrite. This we do here, by virtue of this definition

        # 'trainer' is a global variable
        global trainer

        loss, logits = self(
            batch

        )  # self refers to the model, which in turn acceses the forward method

        # FAH
        if self.model.jrs_do_minibatch_trimming:
            loss = mbtrim.get_adapted_loss_for_trimmed_minibatch(loss, trainer.current_epoch, trainer.max_epochs, 1.0, 0.2, 'linear')

        # Here we can do e.g. mini-batch trimming eg..
        # For now, we just do a usual reduction via 'mean' (otherwise we cannot optimize, loss must be scalar value)
        if self.model.jrs_do_loss_reduction == False:
            loss = torch.mean(loss)
        # ~FAH

        self.log('train_loss', loss)
        # pytorch lightning allows you to use various logging facilities, eg tensorboard with tensorboard we
        # can track and easily visualise the progress of training. In this case

        return {'loss': loss}
        # the training_step method expects either a dictionary or a the loss as a number

    def validation_step(self, batch, batch_nb):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class  wants you to overwrite, in case you want to do validation. This we do here, by virtue of this
        # definition.

        loss, logits = self(batch)
        # FAH if we have disabled loss reduction (in training step), we have to do it here in validation step ...
        # In validation step, it does not has any sense to do NOT a reduction
        if self.model.jrs_do_loss_reduction == False:
            loss = torch.mean(loss)
        # self refers to the model, which in turn accesses the forward method

        # Apart from the validation loss, we also want to track validation accuracy  to get an idea, what the
        # model training has achieved "in real terms".

        labels = batch[2]
        predictions = torch.argmax(logits, dim=1)
        accuracy = (labels == predictions).float().mean()

        self.log('val_loss', loss)
        self.log('accuracy', accuracy)
        # the validation_step method expects a dictionary, which should at least contain the val_loss
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def validation_epoch_end(self, validation_step_outputs):
        # OPTIONAL The second parameter in the validation_epoch_end - we named it validation_step_outputs -
        # contains the outputs of the validation_step, collected for all the batches over the entire epoch.

        # We use it to track progress of the entire epoch, by calculating averages

        avg_loss = torch.stack([x['val_loss']
                                for x in validation_step_outputs]).mean()

        avg_accuracy = torch.stack(
            [x['val_accuracy'] for x in validation_step_outputs]).mean()

        print (f" -> Epoch: {self.current_epoch} -- validation loss: {avg_loss} -- validation accuracy: {avg_accuracy}")

        tensorboard_logs = {'val_avg_loss': avg_loss, 'val_avg_accuracy': avg_accuracy}

        return {
            'val_loss': avg_loss,
            'log': tensorboard_logs,
            'progress_bar': {
                'avg_loss': avg_loss,
                'avg_accuracy': avg_accuracy
            }
        }
        # The training_step method expects a dictionary, which should at least contain the val_loss. We also
        # use it to include the log - with the tensorboard logs. Further we define some values that are
        # displayed in the tqdm-based progress bar.


    def configure_optimizers(self):
        # The configure_optimizers is a (virtual) method, specified in the interface, that the
        # pl.LightningModule class wants you to overwrite.

        # In this case we define that some parameters are optimized in a different way than others. In
        # particular we single out parameters that have 'bias', 'LayerNorm.weight' in their names. For those
        # we do not use an optimization technique called weight decay.

        global global_enable_adafamily


        if not global_enable_adafamily:
            self.optimizer = AdamW(self.parameters(),
                              lr=self.hparams.learning_rate,
                              weight_decay = 1e-4,
                              # FAH Note for NLP training it is better to set the 'epsilon' value smaller !
                              eps=1e-16
                              # args.adam_epsilon  - default is 1e-8.
                              )
        else:

            self.optimizer = AdaFamily(self.parameters(),
                                   lr=self.hparams.learning_rate,
                                   weight_decay = 1e-4,
                                   myu = 0.25,
                                   epsilon_mode = 0,
                                   eps=1e-16,
                                   # Switch these off, as they are also not implemented in 'AdamW'
                                   rectify = False, degenerated_to_sgd = False,
                                   # args.adam_epsilon  - default is 1e-8.
                                   )

        # We also use a scheduler that is supplied by transformers.
        # FAH: For more info see https://stackoverflow.com/questions/60120043/optimizer-and-scheduler-for-bert-fine-tuning
        # FAH: I change the schedule from linear-schedule-with-warmup (where warmup actually is disabled)
        # to a normal schedule with _constant_ learning rate. This is, to ease the later integration into my 'dist_p1' prototype.
        #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
            # Default value in run_glue.py
            # num_training_steps=self.hparams.num_training_steps)
        #return [optimizer], [scheduler]
        #return [optimizer]
        # FAH We decay learning rate by a factor of 0.3 every 4 epochs ..
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 4, gamma = 0.3)
        return [self.optimizer], [self.scheduler]

    def freeze(self) -> None:
        # freeze all layers, except the final classifier layers
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:  # classifier layer
                param.requires_grad = False


        self._frozen = True

    def unfreeze(self) -> None:
        if self._frozen:
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.requires_grad = True

        self._frozen = False

    def on_epoch_start(self):
        """pytorch lightning hook"""
        if self.current_epoch < self.hparams.nr_frozen_epochs:
            self.freeze()

        if self.current_epoch >= self.hparams.nr_frozen_epochs:
            self.unfreeze()

class Data(pl.LightningDataModule):
    # So here we finally arrive at the definition of our data class derived from pl.LightningDataModule.
    #
    # In earlier versions of pytorch lightning  (prior to 0.9) the methods here were part of the model class
    # derived from pl.LightningModule. For better flexibility and readability the Data and Model related parts
    # were split out into two different classes:
    #
    # pl.LightningDataModule and pl.LightningModule
    #
    # with the Model related part remaining in pl.LightningModule
    #
    # This is explained in more detail in this video: https://www.youtube.com/watch?v=L---MBeSXFw

    def __init__(self, *args, **kwargs):
        super().__init__()

        # self.save_hyperparameters()
        if isinstance(args, tuple):
            args = args[0]
        # FAH had to change for pytorch lightning version >= 1.5 otherwise I get error.
        #     See issue at https://digital-wiki.joanneum.at/pages/viewpage.action?pageId=124552994
        # self.hparams = args
        self.hparams.update(vars(args))
        # cf this open issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/3232

        print('args:', args)
        print('kwargs:', kwargs)

        # print(f'self.hparams.pretrained:{self.hparams.pretrained}')

        print('Loading BERT tokenizer')
        print(f'PRETRAINED:{self.hparams.pretrained}')

        A = AutoTokenizer
        self.tokenizer = A.from_pretrained(self.hparams.pretrained)

        print('Type tokenizer:', type(self.tokenizer))

        # This part is easy  we instantiate the tokenizer

        # So this is easy, but it's also incredibly important, e.g. in this by using "bert-base-uncased", we
        # determine, that before any text is analysed its all turned into lower case. This might have a
        # significant impact on model performance!!!
        #
        # BertTokenizer is the tokenizer, which is loaded automatically by AutoTokenizer because it was used
        # to train the model weights of "bert-base-uncased".

    def prepare_data(self):
        # Even if you have a complicated setup, where you train on a cluster of multiple GPUs, prepare_data is
        # only run once on the cluster.

        # Typically - as done here - prepare_data just performs the time-consuming step of downloading the
        # data.

        print('Setting up dataset')

        prefix = 'https://drive.google.com/uc?id='
        id_apps = "1S6qMioqPJjyBLpLVz4gmRTnJHnjitnuV"
        id_reviews = "1zdmewp7ayS4js4VtrJEHzAheSW-5NBZv"

        pathlib.Path('./data').mkdir(parents=True, exist_ok=True)

        # Download the file (if we haven't already)
        if not os.path.exists('./data/apps.csv'):
            gdown.download(url=prefix + id_apps,
                           output='./data/apps.csv',
                           quiet=False)

        # Download the file (if we haven't already)
        if not os.path.exists('./data/reviews.csv'):
            gdown.download(url=prefix + id_reviews,
                           output='./data/reviews.csv',
                           quiet=False)

    def setup(self, stage=None):
        # Even if you have a complicated setup, where you train on a cluster of multiple GPUs, setup is run
        # once on every gpu of the cluster.

        # typically - as done here - setup
        # - reads the previously downloaded data
        # - does some preprocessing such as tokenization
        # - splits out the dataset into training and validation datasets

        # Load the dataset into a pandas dataframe.
        df = pd.read_csv("./data/reviews.csv", delimiter=',', header=0)

        if self.hparams.frac < 1:
            df = df.sample(frac=self.hparams.frac, random_state=0)

        df['score'] -= 1

        # Report the number of sentences.
        print('Number of training sentences: {:,}\n'.format(df.shape[0]))

        # Get the lists of sentences and their labels.
        sentences = df.content.values
        labels = df.score.values

        t = self.tokenizer(
            sentences.tolist(),  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=128,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'  # Return pytorch tensors.
        )

        # Convert the lists into tensors.
        input_ids = t['input_ids']
        attention_mask = t['attention_mask']

        labels = torch.tensor(labels) # .float() if regrssion

        # Print sentence 0, now as a list of IDs. print('Example') print('Original: ', sentences[0])
        # print('Token IDs', input_ids[0]) print('End: Example')

        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_mask, labels)

        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(self.hparams.training_portion * len(dataset))
        val_size = len(dataset) - train_size

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))

        # FAH WARNING: We use _always_ the SAME seed for generating the random split !!
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(global_random_seed))

    def train_dataloader(self):
        # as explained above, train_dataloader was previously part of the model class derived from
        # pl.LightningModule train_dataloader needs to return the a Dataloader with the train_dataset

        return DataLoader(
            self.train_dataset,  # The training samples.
            sampler=RandomSampler(
                self.train_dataset),  # Select batches randomly
            batch_size=self.hparams.batch_size,  # Trains with this batch size.
            # FAH increased num_workers to 4 for a (hopefully) better performance
            num_workers = 4
        )

    def val_dataloader(self):
        # as explained above, train_dataloader was previously part of the model class derived from
        # pl.LightningModule train_dataloader needs to return the a Dataloader with the val_dataset
        xyz = 123

        return DataLoader(
            self.val_dataset,  # The training samples.
            sampler=RandomSampler(self.val_dataset),  # Select batches randomly
            batch_size=self.hparams.batch_size,  # Trains with this batch size.
            shuffle=False,
            # FAH increased num_workers to 4 for a (hopefully) better performance
            num_workers = 4
        )


def main():
    # Two key aspects:

    # - pytorch lightning can add arguments to the parser automatically

    # - you can manually add your own specific arguments.

    # - there is a little more code than seems necessary, because of a particular argument the scheduler
    #   needs. There is currently an open issue on this complication
    #   https://github.com/PyTorchLightning/pytorch-lightning/issues/1038

    # FAH 'trainer' and the others are global variables
    global trainer
    global global_do_minibatch_trimming
    global global_random_seed
    global global_enable_adafamily

    import argparse
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # We use the parts of very convenient Auto functions from huggingface. This way we can easily switch
    # between models and tokenizers, just by giving a different name of the pretrained model.
    #
    # BertForSequenceClassification is one of those pretrained models, which is loaded automatically by
    # AutoModelForSequenceClassification because it corresponds to the pretrained weights of
    # "bert-base-uncased".

    # Similarly BertTokenizer is one of those tokenizers, which is loaded automatically by AutoTokenizer
    # because it is the necessary tokenizer for the pretrained weights of "bert-base-uncased".
    parser.add_argument('--pretrained', type=str, default="distilbert-base-uncased")
    parser.add_argument('--nr_frozen_epochs', type=int, default=5)
    parser.add_argument('--training_portion', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--frac', type=float, default=1)
    # FAH added
    parser.add_argument('--seed', type=int, default=42)
    # if you append '--enable_mbtrim' to command-line, then the variable will be 'True', otherwise 'False
    parser.add_argument('--enable_mbtrim', action='store_true')
    parser.add_argument('--enable_adafamily', action='store_true')


    # FAH Regarding number of epochs for 'freezing' (parameter 'nr_frozen_epochs'), see
    # https://discuss.huggingface.co/t/fine-tune-bert-models/1554/5
    # https://www.reddit.com/r/deeplearning/comments/ndmqm6/why_do_we_train_whole_bert_model_for_fine_tuning/
    # https://discuss.huggingface.co/t/fine-tune-bert-models/1554
    # https://ai.stackexchange.com/questions/23884/why-arent-the-bert-layers-frozen-during-fine-tuning-tasks
    # according to the last link, it may not be necessary to freeze the model layers at all.
    # We freeze them for 20-30 % of the total amount of epochs.
    # That might be a good rule of thumb

    # FAH Regarding 'max_epochs' for fine-tuning, I suppose it is in range 3-10 or so.
    # Often, it seems only to be 3 epochs (which I find a bit on the low side)

    # parser = Model.add_model_specific_args(parser) parser = Data.add_model_specific_args(parser)
    # FAH: 'args' contains all parameters provided by pytorch lightning 'Trainer' objects,
    # _PLUS_ the few additional params (see above) added by us.
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # FAH added
    global_random_seed = args.seed
    global_do_minibatch_trimming = args.enable_mbtrim
    global_enable_adafamily = args.enable_adafamily


    # TODO start: remove this later
    # args.limit_train_batches = 10 # TODO remove this later
    # args.limit_val_batches = 5 # TODO remove this later
    # args.frac = 0.01 # TODO remove this later
    # args.fast_dev_run = True # TODO remove this later
    # args.max_epochs = 2 # TODO remove this later

    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='lightning_logs')

    parser.logger = logger
    # TODO end: remove this later

    # start : get training steps
    d = Data(args)
    d.prepare_data()
    d.setup()


    args.num_training_steps = len(d.train_dataloader()) * args.max_epochs
    # end : get training steps


    dict_args = vars(args)
    m = Model(**dict_args)


    # FAH: We Disable early stopping on validation loss
    #args.early_stop_callback = EarlyStopping('val_loss')

    # FAH: Disable validation sanity steps at begin
    # see https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html?highlight=num_sanity_val_steps#num-sanity-val-steps
    #dict_args['num_sanity_val_steps'] = 0

    # FAH: Do validation only every few epochs (e.g. every 4 epochs)
    #dict_args['check_val_every_n_epoch'] = 4

    # FAH: Enable training on _GPU_ (otherwise, per default the CPU is used)
    # Note the training on the GPU runs (on my machine) roughly 20x faster ... :-)
    enable_gpu_training = True
    if enable_gpu_training:
        # See https://devblog.pytorchlightning.ai/announcing-the-new-lightning-trainer-strategy-api-f70ad5f9857e
        dict_args['accelerator'] = 'gpu'
        dict_args['gpus'] = 1

    trainer = pl.Trainer.from_argparse_args(args)

    # fit the data
    trainer.fit(m, d)

    # %%

if __name__ == "__main__":
    main()

