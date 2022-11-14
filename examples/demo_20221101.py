import deepinv as dinv

dataloader = dinv.datasets.mnist_dataloader(mode='test', batch_size=128, num_workers=4, shuffle=True)

physics = dinv.physics.compressed_sensing(m=100, img_shape=(1,28,28)).to(dinv.device)

model = dinv.models.unet(in_channels=1,
                         out_channels=1,
                         circular_padding=True,
                         compact=3).to(dinv.device)


loss_mc = dinv.loss.MCLoss(physics=physics,
                           metric=dinv.metric.mse(dinv.device))

loss_ei = dinv.loss.EILoss(transform=dinv.transform.Shift(n_trans=2),
                           physics=physics,
                           metric=dinv.metric.mse(dinv.device))

loss_ms = dinv.loss.MeaSplitLoss(physics=physics,
                                 metric=dinv.metric.mse(dinv.device),
                                 split_ratio=0.9)


optimizer = dinv.optim.Adam(model.parameters(),
                            lr=1e-4,
                            weight_decay=1e-8)


dinv.train(model=model,
           train_dataloader=dataloader,
           learning_rate=5e-4,
           physics=physics,
           epochs=500,
           schedule=[400],
           loss_closure=[loss_ms,loss_ei],
           # loss_closure=[loss_sure, loss_rei],
           # loss_closure=[loss_ms, loss_ei],
           loss_weight=[1,1],
           optimizer=optimizer,
           device=dinv.device,
           ckp_interval=250,
           save_path='dinv_ms_ei',
           verbos=True)

