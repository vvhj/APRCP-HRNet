import matplotlib.pyplot as plt
import json
import numpy as np
def readdata(filepath):
    with open(filepath,"r") as f:
        data = json.load(f)
    return data
W32 = readdata("/root/work/hrnet_purn/Purn_v4_output/data2.json")
W48 = readdata("/root/work/hrnet_purn/Purn_v4_output/data2.json")
figure, ax = plt.subplots() 
W32ap = W32["Perf_indicator"]
W48ap = W48["Perf_indicator"]
W32acc = W32["Acc"]
W48acc = W48["Acc"]
num1 = []#[2.5, 2.6, 2.7]
num2 = []#[2.75, 2.85, 2.95]
x = W32["percent"] 
for i in range(len(x)):
    num1.append(W32acc[i]/max(W32acc)+W32ap[i]/max(W32ap))
    num2.append(W48acc[i]/max(W48acc)+W48ap[i]/max(W48ap))
idx32 =  np.argmin(np.array(num1)+np.array(x))
idx48 = np.argmin(np.array(num2)+np.array(x))
print(x[idx32])
print(x[idx48])
#plt.scatter([x[idx32]], [num1[idx32]],color = 'g', s = 75 ,label = 'w32 best purned hrnet')
#plt.scatter([x[idx48]], [num2[idx48]],color = 'y', s = 75 ,label = 'w48 best purned hrnet')

a = x[idx32]
b = num1[idx32]
plt.text(a-0.2,b,"w32 best per:{}".format(a),ha = 'center',va = 'bottom',fontsize=7,color ='r')

a = x[idx48]
b = num1[idx48]
plt.text(a-0.15,b,"w48 best per:{}".format(a),ha = 'center',va = 'bottom',fontsize=7,color ='b')

a = x[int(80/2)]
b = num1[int(80/2)]
plt.text(a,b+0.1,"w32 extreme per:{}".format(a),ha = 'center',va = 'bottom',fontsize=7,color ='r')

a = x[int(78/2)]
b = num1[int(78/2)]
plt.text(a,b-0.1,"w48 extreme per:{}".format(a),ha = 'center',va = 'bottom',fontsize=7,color ='b')

plt.plot(x,num1,label='w32_256x192',linestyle='--',color='r',marker='D')
plt.plot(x,num2,label='w48_384x288',linestyle='--',color='b',marker='o')
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
plt.xlabel('per',font1)
plt.ylabel('eva',font1)
plt.title('Candidate Model Selection',font1)
plt.savefig('squares.png')
#plt.show()
