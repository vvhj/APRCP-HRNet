import matplotlib.pyplot as plt
import json
import numpy as np
def readdata(filepath):
    with open(filepath,"r") as f:
        data = json.load(f)
    return data
W32 = readdata("datav3.json")
W48 = readdata("datav3.json")
figure, ax = plt.subplots()
W32ap = W32["Perf_indicator"]
W48ap = W48["Perf_indicator"]
W32acc = W32["Acc"]
W48acc = W48["Acc"]
num1 = []#[2.5, 2.6, 2.7]
num2 = []#[2.75, 2.85, 2.95]
x = W32["percent"][35:] 
for i in range(len(x)):
    num1.append(W32acc[i+35]/max(W32acc)+W32ap[i+35]/max(W32ap))
    num2.append(W48acc[i+35]/max(W48acc)+W48ap[i+35]/max(W48ap))
idx32 =  np.argmin(np.array(num1)+np.array(x))
idx48 = np.argmin(np.array(num2)+np.array(x))
print(x[idx32])
print(x[idx48])
#plt.scatter([x[idx32]], [num1[idx32]],color = 'g', s = 75 ,label = 'w32 best purned hrnet')
#plt.scatter([x[idx48]], [num2[idx48]],color = 'y', s = 75 ,label = 'w48 best purned hrnet')
plt.plot(x,num1,label='w32_256x192',linestyle='--',color='r',marker='D',markersize = 5)
plt.plot(x,num2,label='w48_384x288',linestyle='--',color='b',marker='o',markersize = 5)
a = x[idx32]
b = num1[idx32]
plt.text(a+0.12,b+0.07,"w32 best per:{}".format(a),ha = 'center',va = 'bottom',fontsize=12,color ='r')
plt.plot(a,b,'k*', markersize=10)

a = x[idx48]
b = num2[idx48]
plt.text(a-0.15,b-0.13,"w48 best per:{}".format(a),ha = 'center',va = 'bottom',fontsize=12,color ='b')
plt.plot(a,b,'k*', markersize=10)

a = x[int(80/2)]
b = num1[int(80/2)]
plt.text(a,b+0.07,"w32 extreme per:{}".format(a),ha = 'center',va = 'bottom',fontsize=12,color ='r')
plt.plot(a,b,'ks', markersize=10)

a = x[int(78/2)]
b = num2[int(78/2)]
plt.text(a-0.2,b-0.13,"w48 extreme per:{}".format(a),ha = 'center',va = 'bottom',fontsize=12,color ='b')
plt.plot(a,b,'ks', markersize=10)


plt.yticks(list(np.array(list(range(21)))/10))  #设置x,y坐标值
plt.xticks(list(np.array(list(range(11)))/10))

plt.tick_params(labelsize=16)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
plt.legend(prop=font1,loc=1)

#plt.grid(axis="y")
plt.xlabel('x',font1)
plt.ylabel('y',font1)
#plt.title('Candidate Model Selection',font1)
plt.savefig('squares.png')
plt.show()
