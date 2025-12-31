from main import IncrementalLearning
methods=['naive','ctive','lwf','ewc','si','cwr','ar1','armh','AsyCLR']
datasets=['cifar10','cifar100']
tipo_data=['NIC','NC']
cnns=['cifarnet','lenet','new']

###
method=methods[-1]
dataset=datasets[0]
tipo=tipo_data[1]
cnn=cnns[0]
model=IncrementalLearning(method,dataset,tipo,cnn)
model.train(2500,pat=10,min_epoch=15,L_R=0.0005,num_epocas=6,early_stop_th=0.001,num_classes_ini=20,num_classes_fin=100,incr_class=10)
