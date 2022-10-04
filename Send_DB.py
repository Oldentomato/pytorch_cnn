from pymongo import MongoClient

mongo_url = "mongodb+srv://Oldentomato:jowoosung123@examplecluster.g7o5t.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(host=mongo_url, port=27017)
db = client['Model_Database']
collection = db['pytorch']


class SendLog_ToMongo():
    def __init__(self, data):
        self.Model_Name = 'Inception_'+str(data.get("experiment_count"))
        self.log_data = {}
        self.data = {'model_name':self.Model_Name,
                'batch_size': data.get("batch_size"),
                'optimizer': 'SGD',
                'learning_rate': data.get("learning_rate"),
                'sgd_momentum': data.get("sgd_momentum"),
                'lr_scheduler_gamma': data.get("lr_scheduler_gamma"),
                'lr_scheduler_step':data.get("lr_scheduler_step"),
                'max_epoch': data.get("epoch"),
                'early_stopped': False,
                'logs':self.log_data,
                'test_acc':0.0}
        collection.insert_one(self.data)

    def on_epoch_end(self, epoch, logs=None, early_stopped=False):
        self.log_data = {
            'epoch' : epoch,
            'loss' : logs.get('loss'),
            'acc' : logs.get('accuracy'),
            'val_loss' : logs.get('val_loss'),
            'val_acc' : logs.get('val_accuracy')
        }
        # collection.update_one({'model_name':self.Model_Name},{"$push":self.log_data})
        collection.update_one({'model_name': self.Model_Name}, {"$set": {"logs":self.log_data}})

    def on_test_end(self, test_acc=None):
        collection.update_one({'model_name':self.Model_Name},{"$set":{"test_acc":test_acc}})