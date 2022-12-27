from pymongo import MongoClient

#encrypted
I6x9Mg7Fznb+6gS7htnrPSGqsu0RYpZZAPLvRZNw1nqIGVr9YFCNjM6jQq5jT06zlRq7PqucDDzVMPefzXUXQpiB6nUIoojujDuetDAUhBwbaCV+DG9KIUwXTs/mPcOyZdAZsuofVwu9Sef0BMnMH+/8BRqj4xo3ZV02quYA63py7yg2CnyZzV0fIWPycYQF8I/zQhQ2w5e/foqpDpNQCshwuI3LI1mwuZIhzsthPAHRUEFkC06F0eWA6PsG1MGFxCz2zBJs9/Kds00Gr+6M+QmYXLPhAhjHFSiS/1vRHFjMzGb6SLm2Ne8/aZN5wvc6EbjO4Em5Nn4qQ8vkEaEOwwgxOkkJ/CrbSILR6H7IHZT0QKDC1AebFKxSLvF0O53tMiO44hG1YelOvJtmP0R77fv8EgLB+1Oj362JAboCrS1nusjEVxeGoiDSjAZABwVazNtfmWckNqKjR7h5sakX5r1yn9hPXdsXhhx6jtS+qtUcq68DcKrZrI1Unq0FmZBIDO48Mg+WvyWKjg/sKPN7XVm4rJUHDklmZzSOf/10PKzqYvh1l4f6ErwPGG7MzvYoa26upkbou/7Zji+q9nvOcAHndZ+Srlgkdq8HZD1moMmd7rnk2gzEVCtPmY/JeQoYDl4gaImDRDxEHVF6dMMe2jad/n1Uvg43nszjsx3lcEg=
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