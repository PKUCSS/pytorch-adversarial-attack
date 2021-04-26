import matplotlib.pyplot as plt  


# The effect of step size

## MNIST

size = [0.0125,0.025,0.05,0.1,0.2,0.4,0.8]
asr = [22.39,81.12,97.47,99.63,99.58,92.71,87.74]

plt.plot(size,asr,marker='o')
plt.xlabel(r"$\alpha$")
plt.ylabel("ASR")
plt.title(r"the effect of step size $alpha$")
plt.savefig("pics/MNIST_size.png")
plt.cla()

## CIFAR10

size = [0.00125,0.0025,0.005,0.01,0.02,0.04,0.08]
asr = [98.38,99.98,100,100,100,99.87,91.72]

plt.plot(size,asr,marker='o')
plt.xlabel(r"$\alpha$")
plt.ylabel("ASR")
plt.title(r"the effect of step size $\alpha$")
plt.savefig("pics/CIFAR_size.png")
plt.cla()

# The effect of step num 

num = list(range(1,21))
asr = [5.95,51.73,82.20,89.84,95.26,97.86,98.86,99.27,99.51,99.63,99.66,99.70,99.75,99.76,99.80,99.80,99.81,99.81,99.81,99.83]
plt.plot(num,asr,marker='o',label = 'MNIST' )

asr = [72.41,98.27,99.95] + [100 for _ in range(17)]
plt.plot(num,asr,marker='o',label = 'CIFAR10' )

plt.xlabel(r"T")
plt.ylabel("ASR")
plt.title(r"the effect of step num T")
plt.legend()
plt.savefig("pics/step_num.png")
plt.cla()

# The effect of decay

decay = [0.01,0.025,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
asr = [97.51, 97.57, 97.81, 97.82,98.18,98.19,98.22, 98.37, 98.36, 98.18, 98.09, 97.96, 97.75]
plt.plot(decay,asr,marker='o',label = 'MNIST' )

asr = [86.86, 86.67, 87.66, 88.27, 89.03, 89.82, 89.89, 90.45, 90.68, 90.75, 90.49, 90.07, 89.43]
plt.plot(decay,asr,marker='o',label = 'CIFAR10' )

plt.xlabel(r"$\mu$")
plt.ylabel("ASR")
plt.title(r"the effect of momentum decay $\mu$")
plt.legend()
plt.savefig("pics/decay.png")
plt.cla()