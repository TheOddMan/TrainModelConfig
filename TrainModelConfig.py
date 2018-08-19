config_TrainingDir = "D:\\Maimy\\CNN\\getRandomImageTest\\train"
config_ValidationDir = "D:\\Maimy\\CNN\\getRandomImageTest\\validation"

config_TrainingDataAmount = 70     #percent
config_ValidationDataAmount = 10   #percent


config_Monte_Carlo_time = 2

config_ImageWidth = 64
config_ImageHeight = 64


config_Epochs = 2           #more than 1000
config_Batch_Size = 32

class ConvLayer:

    def __init__(self,Filters,Filters_Size,Activation,Open_Dropout,Dropout_Value,Open_MaxPooling,MaxPooling_Size):
        self.Filters = Filters
        self.Filters_Size = Filters_Size
        self.Activation = Activation
        self.Open_Dropout = Open_Dropout
        self.Dropout_Value = Dropout_Value
        self.Open_MaxPooling = Open_MaxPooling
        self.MaxPooling_Size = MaxPooling_Size

class DenseLayer:
    def __init__(self,Neurons_Amount,Activation,Open_Dropout,Dropout_Value):
        self.Neurons_Amount = Neurons_Amount
        self.Activation = Activation
        self.Open_Dropout = Open_Dropout
        self.Dropout_Value = Dropout_Value



c1 = ConvLayer(Filters=128,Filters_Size=5,Activation='relu',Open_Dropout='off',Dropout_Value=0.7,Open_MaxPooling='on',MaxPooling_Size=2)
c2 = ConvLayer(Filters=128,Filters_Size=5,Activation='relu',Open_Dropout='off',Dropout_Value=0.7,Open_MaxPooling='on',MaxPooling_Size=2)
c3 = ConvLayer(Filters=128,Filters_Size=5,Activation='relu',Open_Dropout='off',Dropout_Value=0.7,Open_MaxPooling='on',MaxPooling_Size=2)
c4 = ConvLayer(Filters=128,Filters_Size=5,Activation='relu',Open_Dropout='off',Dropout_Value=0.7,Open_MaxPooling='on',MaxPooling_Size=2)

config_convlist = [c1,c2,c3,c4]


d1 = DenseLayer(Neurons_Amount=64,Activation='tanh',Open_Dropout='off',Dropout_Value=0.5)
d2 = DenseLayer(Neurons_Amount=64,Activation='tanh',Open_Dropout='off',Dropout_Value=0.5)
d3 = DenseLayer(Neurons_Amount=64,Activation='tanh',Open_Dropout='off',Dropout_Value=0.5)

config_denselist = [d1,d2,d3]



config_Classes_amount = 5

config_Learning_Rate = 0.0001


config_Save_Model_File_Name = 'Param1'
config_Excel_Result_Name = 'Param1'

