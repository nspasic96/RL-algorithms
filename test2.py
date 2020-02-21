from collections import deque


memory = deque(maxlen = 4)
memory.append(1)
memory.append(2)
memory.append(3)
memory.append(4)
memory.append(5)
#print(memory)


class Parent:
    def __init__(self):
        self.a = 5
        
class Child(Parent):    
    def __init__(self):        
        super(Parent, self).__init__()
        self.x = super(Parent).a
        print(self.x)

ch = Child()
