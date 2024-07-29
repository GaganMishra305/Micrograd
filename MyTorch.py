import numpy as np 
import math
import random

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
        self.grad = 0.0
        self._backward = lambda: None
    
    def __repr__(self):
        return f'Value(data={self.data})'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data + other.data, (self, other), '+')
    
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out =  Value((self.data * other.data), (self, other), '*')
    
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only supporting int or float powers'
        out = Value((self.data ** other), (self, ), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        
        return out
            
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self ):
        return self*(-1)
    
    def __radd__(self, other):
        return self + other
    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n)-1)/(math.exp(2*n)+1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return self - other
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        # utilizing topological sort
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
class Neuron:
    
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        
    def parameters(self):
        return self.w + [self.b]
        
    def __call__(self, x):
        # w*x + b
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b) 
        out = act.tanh()
        return out
       
class Layer:
    
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        
class MLP:
    
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def training(self, x_train, y_train, x_test, y_test, epochs=100, lr=0.01, loss='mse'):
        for k in range(epochs):
            
            # forward propaagtion
            y_pred = [self(x) for x in x_train]
            if loss == 'mse':
                loss = sum(((y_gt - y_pred)**2 for y_gt, y_pred in zip(y_train, y_pred)))
            elif loss == 'log_loss':
                loss = -sum((y_gt * math.log(y_pred) + (1-y_gt) * math.log(1-y_pred)) for y_gt, y_pred in zip(y_train, y_pred))
                
            # backward propagation
            for p in self.parameters():
                p.grad = 0.0  # zeroing the gradients
            loss.backward()
            
            # gradient descent
            for p in self.parameters():
                p.data -= lr * p.grad
                
            # calculating val_loss....loss on unseen data
            y_pred = [self(x) for x in x_test]
            val_loss = 0.0
            if loss == 'mse':
                val_loss = sum(((y_gt - y_pred)**2 for y_gt, y_pred in zip(y_test, y_pred)))
            elif loss == 'log_loss':
                val_loss = -sum((y_gt * math.log(y_pred) + (1-y_gt) * math.log(1-y_pred)) for y_gt, y_pred in zip(y_test, y_pred))
                
            print(f'Epoch {k+1}/{epochs}, Loss: {loss.data:.4f}, Val_Loss: {val_loss:.4f}')
        