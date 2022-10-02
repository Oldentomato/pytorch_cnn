import inception_v3 as v3


mongo_url = "mongodb+srv://Oldentomato:jowoosung123@examplecluster.g7o5t.mongodb.net/Model_Database?retryWrites=true&w=majority"
client = MongoClient(host=mongo_url, port=27017)
db = client['Model_Database']
collection = db['Pytorch']


class SendLog_ToMongo():
    def __init__(self):
        self.Model_Name = 'Inception_1'
        self.log_data = {}
        self.data = {'model_name':self.Model_Name,
                'batch_size': v3.batch_size ,
                'optimizer': 'SGD',
                'sgd_momentum': v3.sgd_momentum,
                'lr_scheduler_gamma': v3.lr_scheduler_gamma,
                'lr_scheduler_step':v3.lr_scheduler_step,
                'max_epoch': v3.epoch,
                'logs':self.log_data}
        self.collection.insert_one(self.data)

    def on_epoch_end(self, epoch, logs=None):
        self.log_data = {
            'epoch' : epoch,
            'loss' : logs.get('loss'),
            'acc' : logs.get('accuracy'),
            'val_loss' : logs.get('val_loss'),
            'val_acc' : logs.get('val_accuracy')
        }
        collection.update_one({'model_name':self.Model_Name},{"$push":self.log_data})