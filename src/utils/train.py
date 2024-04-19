import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .accuracy import cls_accuracy
from src import utils
from src import data_preprocessing


def train_model(loaders, model, session_dir, epochs):
    writer = SummaryWriter(log_dir=session_dir)
    logger = utils.get_root_logger()
    best_model = None

    Acc_best = 0.0
    Loss_train = []
    Loss_validation = []
    Acc_train = []
    Acc_validation = []

    for epoch in range(epochs):
        acc_list = []
        loss_list = []
        train_pbar = tqdm(loaders.train_loader)

        for batch_idx, batch in enumerate(train_pbar):
            x_train, y_train = batch
            x = torch.autograd.Variable(x_train.reshape(-1, 784).cuda())
            y = y_train.cuda()
            predicted_porb, loss = model.train(x, y)
            acc_list.append(
                cls_accuracy(predicted_porb.cpu().detach(), y.cpu().detach())
            )
            loss_list.append(loss.cpu().detach())

            writer.add_scalar("Loss/train", loss_list[-1], epoch)
            writer.add_scalar(
                "Accuracy/train",
                acc_list[-1],
                epoch,
            )
            train_pbar.set_description(
                f"🏋️> Epoch [{str(epoch).zfill(3)}/{str(epochs).zfill(3)}] | Loss {loss_list[-1].item():.5f}|Acc {acc_list[-1]*100} "
            )

        Loss_train.append(torch.mean(torch.tensor(loss_list)))
        Acc_train.append(torch.mean(torch.tensor(acc_list)))

        logger.info(
            f"[Epoch {epoch}]Train Loss is {Loss_train[-1]} , Train Accuracy is {Acc_train[-1]*100}"
        )

        acc_list = []
        loss_list = []
        for x_validation, y_validation in loaders.validation_loader:
            x = torch.autograd.Variable(x_validation.reshape(-1, 784).cuda())
            y = y_validation.cuda()
            predicted_porb, loss = model.inference(x, y)
            acc_list.append(
                cls_accuracy(predicted_porb.cpu().detach(), y.cpu().detach())
            )
            loss_list.append(loss.cpu().detach())
            writer.add_scalar("Loss/validation", loss_list[-1], epoch)
            writer.add_scalar(
                "Accuracy/validation",
                acc_list[-1],
                epoch,
            )

        Loss_validation.append(torch.mean(torch.tensor(loss_list)))
        Acc_validation.append(torch.mean(torch.tensor(acc_list)))
        if Acc_validation[-1] > Acc_best:
            Acc_best = Acc_validation[-1]
            ckpt_name = f"VitPose_Acc{Acc_best:.4f}_epoch{str(epoch).zfill(3)}.pth"
            ckpt_path = os.path.join(session_dir, ckpt_name)
            model.save(ckpt_path)
            best_model = model
            logger.info(f"[Epoch {epoch}] Best Model Has Been Saved")

        logger.info(
            f"[Epoch {epoch}]Validation Loss is {Loss_validation[-1]} , Validation Accuracy is {Acc_validation[-1]*100}"
        )
    acc_list = []
    loss_list = []
    predicted_labels = []
    for x_test, y_test in loaders.test_loader:
        x = torch.autograd.Variable(x_test.reshape(-1, 784).cuda())
        y = y_test.cuda()
        predicted_porb, loss = best_model.inference(x, y)
        loss_list.append(loss.cpu().detach())
        acc_list.append(cls_accuracy(predicted_porb.cpu().detach(), y.cpu().detach()))
        predicted_labels.extend(torch.argmax(predicted_porb.cpu(), dim=1))

    logger.info(
        f"[Test Result] Loss is {torch.mean(torch.tensor(loss_list))} , Test Accuracy is {torch.mean(torch.tensor(acc_list))}"
    )
    data_preprocessing.plot_category_test(
        dataset=loaders.test_loader.dataset,
        pred_label=predicted_labels,
        path="data/visualization",
    )
