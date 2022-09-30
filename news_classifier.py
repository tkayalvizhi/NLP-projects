from argparse import Namespace

import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from news_vocabulary_vectorizer_dataset import NewsDataset
from utilities import *


# --------------------------------------------------------------------------------
#
#              SINGLE LAYERED PERCEPTRON
#
# --------------------------------------------------------------------------------


class Perceptron(nn.Module):
    def __init__(self, num_features):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out


# --------------------------------------------------------------------------------
#
#              MULTILAYERED PERCEPTRON
#
# --------------------------------------------------------------------------------


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(MultilayerPerceptron, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        intermediate = F.relu(self.fc1(x_in))
        output = self.fc2(intermediate)

        if apply_softmax:
            output = F.softmax(output, dim=1)

        return output


# --------------------------------------------------------------------------------
#
#                        TRAINING AND TESTING NETWORK
#
# --------------------------------------------------------------------------------


def run_epoch(dataset: NewsDataset, model: Perceptron, train_state: dict,
              args: Namespace, split='train'):
    dataset.set_split(split)
    data_loader = DataLoader(dataset, args.batch_size, shuffle=True)
    p_bar = tqdm_notebook(data_loader, desc=f'split={split}, epoch={train_state["epoch_index"]}', leave=False)
    running_loss = []
    y_true = []
    y_pred = []

    for batch_index, batch_dict in enumerate(p_bar):
        inputs = moveTo(batch_dict['x_data'], args.device)
        labels = moveTo(batch_dict['y_target'], args.device)

        y_hat = model(inputs, apply_sigmoid=True)

        loss = args.loss_func(y_hat.float(), labels.float())

        running_loss.append(loss.item())

        if model.training:
            args.optimizer.zero_grad()
            loss.backward()
            args.optimizer.step()

        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend((y_hat.detach() > 0.5).float().cpu().numpy())

    train_state[split + '_loss'].append(round(np.mean(running_loss), 4))
    train_state[split + '_acc'].append(round(args.score_func(y_true, y_pred), 4))


def train_network(model: Perceptron, dataset: NewsDataset, train_state: dict, args: Namespace):
    print("Classifier in training...")
    model.to(args.device)

    for epoch in tqdm_notebook(range(args.num_epochs)):
        if train_state['stop_early']:
            print('stopping early...')
            break
        train_state['epoch_index'] = epoch
        model = model.train()
        run_epoch(dataset, model, train_state, args, split='train')

        model = model.eval()
        run_epoch(dataset, model, train_state, args, split='val')

        update_train_state(args, model, train_state)

    print('Classifier training done.')
    return train_state


def test_network(model: Perceptron, dataset: NewsDataset, train_state: dict, args: Namespace):
    model = model.eval()

    dataset.set_split('test')
    data_loader = DataLoader(dataset, args.batch_size)
    p_bar = tqdm_notebook(data_loader, desc=f'split=test, epoch={train_state["epoch_index"]}', leave=False)

    running_loss = []
    y_true = []
    y_pred = []

    for batch_index, batch_dict in enumerate(p_bar):
        inputs = moveTo(batch_dict['x_data'], args.device)
        # labels = moveTo(batch_dict['y_target'], args.device)

        y_hat = model(inputs, apply_sigmoid=True)

        # loss = args.loss_func(y_hat.float(), labels.float())
        # running_loss.append(loss.item())
        # y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend((y_hat.detach() > 0.5).float().cpu().numpy())

    # train_state['test_loss'] = round(np.mean(running_loss), 4)
    # train_state['test_acc'] = round(args.score_func(y_true, y_pred), 4)

    return y_pred


# --------------------------------------------------------------------------------
#
#                               MAIN
#
# --------------------------------------------------------------------------------

if __name__ == "__main__":

    args = Namespace(
        frequency_cutoff=25,

        # Data and Path information
        news_csv='data/fake_news/preprocessed_LITE.csv',
        save_dir='model_storage/fake_news/',
        # A PTH file is a machine learning model created using PyTorch
        model_state_file='model.pth',
        vectorizer_file='news_vectorizer.json',
        train_state_file='train_state.json',

        # Model Hyper-parameters
        loss_func=nn.BCELoss(),
        score_func=accuracy_score,

        # Training Hyper-parameters
        batch_size=128,
        early_stopping_criteria=5,
        learning_rate=0.001,
        num_epochs=25,
        seed=42,

        # Runtime options
        catch_keyboard_interrupt=True,
        cuda=True,
        expand_filepaths_to_save_dir=True,
        reload_from_files=False,

    )
    # Expand file paths
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
        args.train_state_file = os.path.join(args.save_dir, args.train_state_file)

        print("Expanded file paths:")
        print(f"\t{args.vectorizer_file}")
        print(f"\t{args.model_state_file}")
        print(f"\t{args.train_state_file}")

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    print(f"Using CUDA: {args.cuda}")

    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Set seed for reproducibility
    set_seed_everywhere(args.seed, args.cuda)

    # handle dirs
    handle_dirs(args.save_dir)

    # --------------------------------------------------------------------------------
    #                                 INITIALIZATION
    # --------------------------------------------------------------------------------

    if args.reload_from_files:
        # Create dataset using class method
        print("Loading dataset and vectorizer...")
        dataset = NewsDataset.load_dataset_and_load_vectorizer(args.news_csv, args.vectorizer_file)
    else:
        print("Loading dataset and creating vectorizer...")
        dataset = NewsDataset.load_dataset_and_make_vectorizer(args.news_csv)
        dataset.save_vectorizer(args.vectorizer_file)

    print("Dataset and vectorizer loaded")

    vectorizer = dataset.get_vectorizer()

    print("Instantiating classifier...")
    classifier = Perceptron(num_features=len(vectorizer.title_vocab) + len(vectorizer.text_vocab))

    # --------------------------------------------------------------------------------
    #                                 TRAINING LOOP
    # --------------------------------------------------------------------------------
    classifier.to(args.device)

    loss_func = args.loss_func
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    args.optimizer = optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)
    args.scheduler = scheduler
    train_state = make_train_state(args)

    train_state = train_network(classifier, dataset, train_state, args)

    save_train_state(train_state, args)

    print(train_state)

    classifier.load_state_dict(torch.load(train_state['model_filename']))
    classifier = classifier.to(args.device)

    prediction = test_network(classifier, dataset, train_state, args)
    print(prediction)
