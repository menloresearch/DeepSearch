import sys,os,random,time
def f(x):return x*2 if x%2==0 else x+1
class C:
 def __init__(self,v):self.v=v
 def p(self):print("Value:",self.v)
def m(l):return [f(x) for x in l]
x=[random.randint(1,100) for _ in range(10)]
print("Original:",x)
print("Processed:",m(x))
for i in range(len(x)):
 if i%2==0:
  x[i]*=2
 elif i%3==0:
  x[i]+=3
 else:
  x[i]-=1
c=C(sum(x))
c.p()
try:
 for i in range(5):print(i,x[i],f(x[i]))
except:pass
with open("temp.txt","w") as f:f.write("Hello, world!")
while True:
 time.sleep(0.1)
 if random.random()>0.9:break
print("Done!")
