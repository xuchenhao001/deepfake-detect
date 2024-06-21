import lightning as L
import torch
from torchmetrics.functional.classification import accuracy
from transformers import DetrForObjectDetection


class DetectModule(L.LightningModule):
    def __init__(
            self,
            optimizer_args: dict,
            num_classes: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        # config for the optimizer and scheduler
        self.optimizer_args = optimizer_args
        self.num_classes = num_classes

        id2label = {0: 'fake', 1: 'real'}
        label2id = {'fake': 0, 'real': 1}
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            num_queries=5,
        )

    def forward(self, images, labels=None):
        # images: (batch_size, num_channel, height, width)
        # apply default mask (batch_size, height, width) where all values set to 1
        mask_shape = (images.shape[0], images.shape[2], images.shape[3])
        default_pixel_mask = torch.ones(mask_shape, device=self.device)

        model_output = self.model(pixel_values=images, pixel_mask=default_pixel_mask, labels=labels)
        return model_output

    def training_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "train")

    def validation_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "val")

    def test_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "test")

    def predict_step(self, batched_inputs, batch_idx):
        images, label = batched_inputs
        model_output = self.model(images)
        print(f"Debug model_output: {model_output}")

    def evaluate(self, batch, stage):
        x, y = batch

        # Convert y into the format expected by DETR.
        #
        # labels (`List[Dict]` of len `(batch_size,)`, *optional*):
        #   Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
        #   following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
        #   respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
        #   in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
        labels = []
        for img_labels, img_boxes in zip(y["labels"], y["boxes"]):
            labels.append({
                "class_labels": img_labels,
                "boxes": img_boxes,
            })
        detr_output = self(x, labels)  # output class: DetrObjectDetectionOutput
        loss = detr_output.loss

        # DETR output logit:
        # logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
        #             Classification logits (including no-object) for all queries.

        # TODO: How can logit be converted to a classification logit?
        # [Option 1] pick the maximum logit among queries, get (batch_size, num_classes + 1)
        # my_pred = torch.max(detr_output.logits[:, :, :2], dim=1)[0]
        # get final pred among classes, (batch_size, 1)
        # my_pred = torch.max(my_pred, dim=1, keepdim=True)[1]

        # [Option 2] voting by respective queries
        my_pred = torch.max(detr_output.logits[:, :, :2], dim=2)[1]  # (batch_size, num_queries)
        # print(f"Debug my_pred 1: {my_pred}")
        # print(f"Debug my_pred 1: {my_pred.shape}")
        my_pred = torch.sum(my_pred, dim=1) / my_pred.shape[1]  # (batch_size, 1)
        # print(f"Debug my_pred 2: {my_pred}")
        # print(f"Debug vote_counts: {my_pred.shape}")
        my_pred = torch.where(my_pred >= 0.5, 1, 0)  # (batch_size)
        my_pred = my_pred.unsqueeze(1)  # (batch_size, 1)
        # print(f"Debug my_pred 3: {my_pred}")
        # print(f"Debug my_pred 4: {my_pred.shape}")

        # compute accuracy
        # https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html
        acc1 = accuracy(my_pred, y["labels"], task="multiclass", num_classes=self.num_classes, top_k=1)

        # log every metric
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_acc1', acc1, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # https://github.com/roboflow/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb
        lr = float(self.optimizer_args["lr"])
        lr_backbone = float(self.optimizer_args["lr_backbone"])
        lr_weight_decay = float(self.optimizer_args["lr_weight_decay"])

        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=lr, weight_decay=lr_weight_decay)
