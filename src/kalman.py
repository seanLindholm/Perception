import numpy as np

class Kalman():
    def __init__(self):
         # Initial State (in left image) 
        self.state = np.array([[0],  # x pos
                        [0],  # x vel
                        [0],  # x accel
                        [0],  # y pos
                        [0],  # y vel
                        [0]]) # y accel

        # Initial Uncertainty (assumed large)
        self.P = np.identity(6, dtype=float) * 1000

        # External motion (assumed environment/camera is stationary)
        self.u = np.array([[0],[0],[0],[0],[0],[0]])

        # Transition matrix (x = x0 + vt + 1/2 at^2, etc. )
        self.F = np.array([[1, 1, 0.5, 0, 0,   0],
                           [0, 1,   1, 0, 0,   0],
                           [0, 0,   1, 0, 0,   0],
                           [0, 0,   0, 1, 1, 0.5],
                           [0, 0,   0, 0, 1,   1],
                           [0, 0,   0, 0, 0,   1]])

        # Observation matrix (only get measured position)
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

        # Measurement Uncertainty
        self.R = np.array([[1, 0], [0, 1]])

        # Identity Matrix
        self.I = np.identity(6, dtype=float)

    def update(self,x,Z):
        self.Y = Z - self.H @ x
        S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ np.linalg.pinv(S)
        X_new = x + self.K @ self.Y
        self.P = (self.I - self.K @ self.H) @ self.P
        return X_new
        
    def predict(self,x):
        X_new = self.F @ x + self.u
        self.P = self.F @ self.P @ self.F.T
        return X_new  
    
    def reset(self):
        self.__init__()
   