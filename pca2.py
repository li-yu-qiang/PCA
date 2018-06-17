# 用PCA扣除光谱中7000埃~9000埃之间的天光谱线
# 选取一定数量的光谱作为样本
# 对样本做PCA，找到天光线的主成分
# 用天光线的主成分，对每个光谱进行减天光
# 2018.6.97
# 李郁强

from sklearn import datasets
from sklearn.decomposition import RandomizedPCA
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

rawData = datasets.load_files("data_folder")


def smooth(flux, ave):
    for i in range(2591, 3820):
        sum = 0
        for j in range(23):
            sum = sum + flux[i - j] + flux[i + j]
            ave[i] = (sum - flux[i]) / 55
    new_ave = ave
    return new_ave


X = rawData.data
y = rawData.filenames
# print(y.shape)

hdu = fits.open(y[0])
lgλ0 = np.array(hdu[0].header['coeff0'])
lgΔλ = np.array(hdu[0].header['coeff1'])
N = np.array(hdu[0].header['naxis1'])
λn = np.array([i for i in range(N)])
wave = 10 ** (lgλ0 + lgΔλ * λn)

last_wave = np.zeros((500, 1229))
last_err = np.zeros((500, 1229))
last_flux = np.zeros((500, 1229))
last_new_ave = np.zeros((500, 1229))
daylight = np.zeros((500, 1229))

for i in range(500):
    hdu = fits.open(y[i])
    flux = np.array(hdu[0].data[0])
    err = np.array(hdu[0].data[2])
    ave = np.zeros(err.shape[0])
    last_wave[i] = wave[2591:3820]
    last_err[i] = err[2591:3820]
    new_ave = smooth(flux, ave)
    last_flux[i] = flux[2591:3820]
    last_new_ave[i] = new_ave[2591:3820]
    daylight[i] = last_flux[i] - last_new_ave[i]

x = daylight.T
y = wave[2591:3820]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
pca = RandomizedPCA(n_components=50, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.fit(X_train_pca, y_train.astype('int'))
y_predict = rbf_svc.predict(X_test_pca)

aim = fits.open('spSpec-51612-0280-152.fit')

lgλ0 = np.array(aim[0].header['coeff0'])
lgΔλ = np.array(aim[0].header['coeff1'])
N = np.array(aim[0].header['naxis1'])
λn = np.array([i for i in range(N)])

wave = wave[2591:3820]
err = err[2591:3820]
for i in range(flux.shape[0]):
    if (flux[i]>10 or flux[i]<2.5):
        flux[i] = (flux[i - 2] + flux[i - 1] + flux[i + 1] + flux[i + 2] + flux[i + 3] + flux[i - 3]) / 6
new_ave = smooth(flux, ave)
flux = flux[2591:3820]
new_ave = new_ave[2591:3820]
daylight = flux - new_ave

rbf_svc.fit(daylight.reshape(-1, 1), wave.reshape(-1, 1).astype('int'))
y_predict = rbf_svc.predict(daylight.reshape(-1, 1))
wave_pred = y_predict[np.argsort(y_predict)]

true_daylight = np.zeros(1229)
for i in range(1229):
    for j in range(1229):
        if (int(wave[i]) == int(wave_pred[j])):
            true_daylight[i] = daylight[j]

# plt.plot(wave_pred, true_daylight, label="天光",color='yellow', linewidth=0.2)#天光
plt.plot(wave, flux, label="流量", color='black', linewidth=0.2)
plt.plot(wave, flux - err, label="误差", color='red', linewidth=0.2)
plt.plot(wave, flux + err, label="误差", color='red', linewidth=0.2)
plt.plot(wave, flux - true_daylight, label='实际', color='green', linewidth=0.2)
plt.legend()
plt.show()
