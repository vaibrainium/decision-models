import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\mablab\\AppData\\Local\\Programs\\Python\\Python38\\ffmpeg\\bin\\ffmpeg.exe'


np.random.seed(5)

# # Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# writervideo = animation.FFMpegWriter(fps=140)
#
#

class Anim():
    def __init__(self, **kw):
        pass

    def ReLU(self,input):
        return input * (input >= 0)

    def Pre_aniRaceModel(self):
        self.fig, self.ax = plt.subplots(1, 1)#, figsize=(8, 8))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.axhline(0, color='white', lw=2,ls='--')

    def RaceModel(self):
        self.frame = 0
        self.coherence = 50
        self.drift_gain = 3*10e-5
        self.drift_gain2 = 3*10e-5
        self.start_point = 0
        self.start_point2 = 0
        self.drift_rate = self.ReLU(self.coherence) * self.drift_gain
        self.drift_rate2 = self.ReLU(-self.coherence) * self.drift_gain2
        self.drift_std = 10e-3
        self.Ubound = 1
        self.Lbound = -1
        self.Pointer = []
        self.Pointer2 = []
        self.Time = []
        while True:
            self.Time.append(self.frame)
            if self.frame == 0:
                self.Pointer.append(self.start_point)
                self.Pointer2.append(self.start_point)
            else:
                Step = self.drift_rate + np.random.normal(loc=0, scale=self.drift_std)
                self.Pointer.append(self.ReLU(self.Pointer[-1] + Step))
                Step2 = self.drift_rate2 + np.random.normal(loc=0, scale=self.drift_std)
                self.Pointer2.append(self.ReLU(self.Pointer2[-1] + Step2))

            self.frame += 1
            if (np.abs(self.Pointer[-1]) > 1) or (np.abs(self.Pointer2[-1]) > 1):
                break

    def animateRaceModel(self):
        self.RaceModel()
        self.Pre_aniRaceModel()
        self.aniRaceModel = animation.FuncAnimation(self.fig, self.updateRaceModel, frames=self.frame, repeat=False)
        # plt.show()

    def updateRaceModel(self, i):
        self.ax.clear()
        self.ax.set_xlim([0, 100])
        self.ax.set_ylim([0, 1.15])
        self.ax.axhline(0, color='white', lw=2, ls='--')
        self.ax.axhline(1, color='white', lw=2, ls='-')
        self.ax.plot([0, 0], [-1, 1], color='white', lw=5, ls='-')
        self.ax.set_xticks([])
        self.ax.set_yticks([0,1]);
        self.ax.set_yticklabels(('','Decision Bound'), color='white')
        self.ax.set_xlabel('Time -->', color='white')

        self.ax.plot(self.Time[:i + 1], self.Pointer[:i + 1], color='green', linewidth=3)
        self.ax.scatter(self.Time[i], self.Pointer[i], s=60, c=[[255 / 255, 20 / 255, 147 / 255]])
        self.ax.plot(self.Time[:i + 1], self.Pointer2[:i + 1], color='red', linewidth=3)
        self.ax.scatter(self.Time[i], self.Pointer2[i], s=60, c=[[255 / 255, 20 / 255, 147 / 255]])
        plt.tight_layout()



    def Pre_aniDDM(self):
        self.fig, self.ax = plt.subplots(1, 1)#, figsize=(15, 5))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black');
        self.ax.axhline(0, color='white', lw=2, ls='--')

    def DDM(self):
        self.frame=0
        self.coherence = 50
        self.drift_rate = 0.05
        self.start_point = 0
        self.drift_std = 0.08
        self.Ubound = 1
        self.Lbound = -1
        self.Pointer = []
        self.Time = []
        while True:
            self.Time.append(self.frame)
            if self.frame==0:
                self.Pointer.append(self.start_point)
            else:
                Step = (self.coherence/100)*self.drift_rate + np.random.normal(loc=0,scale=self.drift_std)
                self.Pointer.append(self.Pointer[-1] + Step)

            self.frame += 1
            if np.abs(self.Pointer[-1]) > 1:
                break


    def animateDDM(self):
        self.DDM()
        self.Pre_aniDDM()
        self.aniDDM = animation.FuncAnimation(self.fig, self.updateDDM, frames=self.frame, repeat = False)
        # plt.show()

    def updateDDM(self,i):
        self.ax.clear()
        self.ax.set_xlim([0,100])
        self.ax.set_ylim([-1.15,1.15])
        self.ax.axhline(0, color='white', lw=2, ls='--')
        self.ax.axhline(1, color='white', lw=2, ls='-')
        self.ax.axhline(-1, color='white', lw=2, ls='-')
        self.ax.plot([0,0],[-1,1], color='white', lw=5, ls='-')
        self.ax.set_xticks([])
        self.ax.set_yticks([-1,1]);  self.ax.set_yticklabels(('Lower Bound', 'Upper Bound'),color='white')
        self.ax.set_xlabel('Time -->',color='white')

        self.ax.plot(self.Time[:i+1],self.Pointer[:i+1], color='green',linewidth=3)
        self.ax.scatter(self.Time[i],self.Pointer[i],s = 60,c=[[255/255,20/255,147/255]])
        plt.tight_layout()



    def Pre_aniUGM(self):
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.patch.set_facecolor('black')
        self.ax[0].set_facecolor('black'); self.ax[1].set_facecolor('black'); self.ax[2].set_facecolor('black')
        self.ax[0].axhline(0, color='white', lw=2,ls='--'); self.ax[2].axhline(0, color='white', lw=2,ls='--')

    def UGM(self):
        self.frame=0
        self.coherence = 30
        self.drift_rate = 0.05
        self.start_point = 0
        self.drift_std = 0.08
        self.Ubound = 1
        self.Lbound = -1
        self.Pointer = []
        self.Time = []
        self.Urgency = []; self.Urgency_slope = 0.02; self.UrgencyPointer = []


        while True:
            self.Time.append(self.frame)
            if self.frame==0:
                self.Pointer.append(self.start_point)
                self.Urgency.append(1 + (np.array(self.Time[self.frame]) *self.Urgency_slope))
                self.UrgencyPointer.append(self.Pointer[-1]*self.Urgency[-1])
            else:
                Step = (self.coherence/100)*self.drift_rate + np.random.normal(loc=0,scale=self.drift_std)
                self.Pointer.append(self.Pointer[-1] + Step)
                self.Urgency.append(1 + (np.array(self.Time[self.frame]) *self.Urgency_slope))
                self.UrgencyPointer.append(self.Pointer[-1]*self.Urgency[-1])
            self.frame += 1


            if np.abs(self.UrgencyPointer[-1]) > 1:
                break

    def animateUGM(self):
        self.UGM()
        self.Pre_aniUGM()
        self.aniUGM = animation.FuncAnimation(self.fig, self.updateUGM, frames=self.frame, repeat = False)
        # plt.show()

    def updateUGM(self,i):
        self.ax[0].clear(); self.ax[1].clear(); self.ax[2].clear()
        self.ax[0].set_xlim([0,100])
        self.ax[0].set_ylim([-1.15,1.15])
        self.ax[0].axhline(0, color='white', lw=2, ls='--')
        self.ax[0].axhline(1, color='white', lw=2, ls='-')
        self.ax[0].axhline(-1, color='white', lw=2, ls='-')
        self.ax[0].plot([0,0],[-1,1], color='white', lw=5, ls='-')
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([-1,1]);  self.ax[0].set_yticklabels(('Lower Bound', 'Upper Bound'),color='white')
        self.ax[0].set_xlabel('Time -->',color='white')


        self.ax[1].axhline(np.min(self.Urgency), color='white', lw=2, ls='-')
        self.ax[1].axvline(0, color='white', lw=2, ls='-')
        self.ax[1].set_xlim([0, len(self.Urgency)+1])
        self.ax[1].set_ylim([np.min(self.Urgency), np.max(self.Urgency)])
        self.ax[1].set_xlabel('Time -->', color='white')
        self.ax[1].set_ylabel('Urgency', color='white')

        self.ax[2].set_xlim([0,100])
        self.ax[2].set_ylim([-1.15,1.15])
        self.ax[2].axhline(0, color='white', lw=2, ls='--')
        self.ax[2].axhline(1, color='white', lw=2, ls='-')
        self.ax[2].axhline(-1, color='white', lw=2, ls='-')
        self.ax[2].plot([0,0],[-1,1], color='white', lw=5, ls='-')
        self.ax[2].set_xticks([])
        self.ax[2].set_yticks([-1,1]);  self.ax[2].set_yticklabels(('Lower Bound', 'Upper Bound'),color='white')
        self.ax[2].set_xlabel('Time -->',color='white')

        self.ax[0].plot(self.Time[:i+1],self.Pointer[:i+1], color='green',linewidth=3)
        self.ax[0].scatter(self.Time[i],self.Pointer[i],s = 60,c=[[255/255,20/255,147/255]])


        self.ax[1].plot(self.Time[:i+1],self.Urgency[:i+1], color='yellow',linewidth=2)

        self.ax[2].plot(self.Time[:i+1],self.UrgencyPointer[:i+1], color='green',linewidth=3)
        self.ax[2].scatter(self.Time[i],self.UrgencyPointer[i],s = 60,c=[[255/255,20/255,147/255]])
        plt.tight_layout()



    def Pre_aniLCA(self):
        self.fig, self.ax = plt.subplots(1, 1)#, figsize=(8, 8))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.axhline(0, color='white', lw=2,ls='--')

    def LCA(self):
        self.frame = 0
        self.coherence = 50
        self.drift_gain = 10e-5
        self.drift_gain2 = 5*10e-6
        self.start_point = 0
        self.start_point2 = 0
        self.drift_rate = self.ReLU(self.coherence) * self.drift_gain
        self.drift_rate2 = self.ReLU(-self.coherence) * self.drift_gain2
        self.leak = 0.01
        self.lateral_inhibition = 0.02
        self.drift_std = 10e-2
        self.Ubound = 1
        self.Lbound = -1
        self.Pointer = []
        self.Pointer2 = []
        self.Time = []
        while True:
            self.Time.append(self.frame)
            if self.frame == 0:
                self.Pointer.append(self.start_point)
                self.Pointer2.append(self.start_point2)
            else:
                Step = self.drift_rate + np.random.normal(loc=0, scale=self.drift_std) - (self.leak*self.Pointer[-1]) - (self.lateral_inhibition*self.Pointer2[-1])
                self.Pointer.append(self.ReLU(self.Pointer[-1] + Step))
                Step2 = self.drift_rate2 + np.random.normal(loc=0, scale=self.drift_std) - (self.leak*self.Pointer2[-1]) - (self.lateral_inhibition*self.Pointer[-1])
                self.Pointer2.append(self.ReLU(self.Pointer2[-1] + Step2))

            self.frame += 1
            print(self.Pointer[-1], self.Pointer2[-1])
            if (np.abs(self.Pointer[-1]) > 1) or (np.abs(self.Pointer2[-1]) > 1):
                break


    def animateLCA(self):
        self.LCA()
        self.Pre_aniLCA()
        self.aniLCA = animation.FuncAnimation(self.fig, self.updateLCA, frames=self.frame, repeat=False)
        # plt.show()

    def updateLCA(self, i):
        self.ax.clear()
        self.ax.set_xlim([0, 100])
        self.ax.set_ylim([0, 1.15])
        self.ax.axhline(0, color='white', lw=2, ls='--')
        self.ax.axhline(1, color='white', lw=2, ls='-')
        self.ax.plot([0, 0], [-1, 1], color='white', lw=5, ls='-')
        self.ax.set_xticks([])
        self.ax.set_yticks([0,1]);
        self.ax.set_yticklabels(('','Decision Bound'), color='white')
        self.ax.set_xlabel('Time -->', color='white')

        self.ax.plot(self.Time[:i + 1], self.Pointer[:i + 1], color='green', linewidth=3)
        self.ax.scatter(self.Time[i], self.Pointer[i], s=60, c=[[255 / 255, 20 / 255, 147 / 255]])
        self.ax.plot(self.Time[:i + 1], self.Pointer2[:i + 1], color='red', linewidth=3)
        self.ax.scatter(self.Time[i], self.Pointer2[i], s=60, c=[[255 / 255, 20 / 255, 147 / 255]])
        plt.tight_layout()






# f = r"C:\Users\mablab\Desktop\animation_DDM.mp4"
# writervideo = animation.FFMpegWriter(fps=20)

# # GIF Format
# writervideo = animation.PillowWriter(fps=20)
# f = r"C:\Users\mablab\Desktop\animation_DDM.gif"

A = Anim()
A.animateRaceModel()
f = r"D:\RESEARCH PROJECTS\Decision-Models\scripts\media\animation_Race.mp4"
writervideo = animation.FFMpegWriter(fps=7)
A.aniRaceModel.save(f, writer=writervideo)

A = Anim()
A.animateDDM()
f = r"D:\RESEARCH PROJECTS\Decision-Models\scripts\media\animation_DDM.mp4"
writervideo = animation.FFMpegWriter(fps=7)
A.aniDDM.save(f, writer=writervideo)

A = Anim()
A.animateUGM()
f = r"D:\RESEARCH PROJECTS\Decision-Models\scripts\media\animation_UGM.mp4"
writervideo = animation.FFMpegWriter(fps=7)
A.aniUGM.save(f, writer=writervideo)

A = Anim()
A.animateLCA()
f = r"D:\RESEARCH PROJECTS\Decision-Models\scripts\media\animation_LCA.mp4"
writervideo = animation.FFMpegWriter(fps=7)
A.aniLCA.save(f, writer=writervideo)