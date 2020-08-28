import numpy as np

class LogisticRegression_GRAD():
    def __init__(self):
        pass
    
    def fit(self, X, y, epochs=30, learning_rate=0.02):
        #Custo
        self.custos = np.array([])
        
        #Bias
        bias = np.ones((X.shape[0], 1))
        X = np.hstack((bias,X))
        self.w = np.ones(X.shape[1])
        print(self.w)
            
        for i in range(0, epochs):
            #Y predito
            y_pred = np.array([])
            for j in range(0, X.shape[0]):
                y_pred = np.append(y_pred, 1 / (1 + np.exp(np.sum((-1) * np.transpose(self.w) * X[j]))))
            
            #Calculo do Somatório(ei * xi)
            exi = 0     
            for j in range(0, X.shape[0]):
                exi += (y[j] - y_pred[j]) * X[j] #Correção no gradiente
            exi_n = (exi/X.shape[0])
            
            #Atualização dos pesos
            self.w = self.w + (learning_rate * exi_n)
            
            #Calcula e salva custo
            custo = np.sum((-1) * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)) / X.shape[0] * 2
            self.custos = np.append(self.custos, custo)

        print("Coeficientes: " + str(self.w))
                          
    def predict(self, X):
        result = np.array([])
        y_pred = np.array([])
        for j in range(0, X.shape[0]):
            y_pred = np.append(y_pred, 1 / (1 + np.exp((-1) * self.w[0] + np.sum((-1) * np.transpose(self.w[1:]) * X[j]))))
                
        for i in y_pred:
            if i < 0.5:
                result = np.append(result, 0)
            else:
                result = np.append(result, 1)
                      
        return result






    