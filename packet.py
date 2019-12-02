"""
Dependencies:
	PyTorch

CAPITALS MAKE A BIG DIFFERENCE IN VARIABLE MEANINGS!

note the data generated has following shape to interface with PyTorch:
(packets,timesteps,channels))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random,time
from random import randint as rint

class Zoom():
	def __init__(self,dataRange):
		self.defaultRange = dataRange

	def zoom(self,level,shift=0):
		start = self.defaultRange[0]
		end = self.defaultRange[1]
		length = end - start
		increment = (length/10)/2
		zoomRange = (start+increment*level+shift,end-increment*level+shift,)
		return zoomRange

class DataGenerator(object):
	def __init__(self,Sigma):
		"""
			inputs:
					N --[int] number of times to sample each packet
					Sigma --[object] frame def:
						{
							'type':'Signatures' or 'MetaSignature',
							'signature': [object]
						}
						the signature of the frame is either predefined signatures
						or a "meta signature" which is used to define how to auto generate 
						some waves 
		"""
		if Sigma['type'] is 'MetaSignature':
			self.Sigma = self.metaToSigs(Sigma)
		else:
			self.Sigma = Sigma

	def metaToSigs(self,metaSig):
		"""
			inputs:
					metaSig --[dict] frame def: 
						{
							'numPackets':int,  #how many different packets to sample (ie number of classes)
							'numChannels':int, #how many channels should each packet have
							'numSoids':int,    #how many sinusoids to superpose for each channel 
							'dead': float,     #set random parts of generated signal to 0
							'epsilon':float,   #scalar used to control amplitude of noise,
							'timesteps': int   #how many timesteps are in each packet,
							'range':tuple(start:int,end:int),
							'impose': bool,    #whether or not to superpose other signals with the signal we want
							'always_impose': bool, # if we should always impose spurious signals
							'sub_sample': float #generate a sample in a smaller range that is a sub range of range
												# and se the rest to 0
						}
		"""
		packetSigs = []
		t = metaSig['timesteps']
		self.timesteps = t
		self.dead = metaSig['dead']
		self.epsilon = metaSig['epsilon']
		self.range = metaSig['range']
		self.numPackets = metaSig['numPackets']
		self.numChannels = metaSig['numChannels']
		self.numSoids = metaSig['numSoids']
		self.impose = metaSig['impose']
		self.always_impose = metaSig['always_impose']
		self.sub_sample = metaSig['sub_sample']

		for i in range(0,metaSig['numPackets']):
			#toplevel packet data
			packetSig = self.packetSig()
			packetSigs.append(packetSig);
		
		return packetSigs

	def randWaveSig(self):
		"""
			generates a reandom signature for a wave
		"""
		randSig = {
			'phase':random.randint(-300,300)/300,
			'freq':random.uniform(0.5,17),
			'amp':random.uniform(.1,1)*(-1,1)[random.randint(0,1)],
			'op':('+','*')[random.randint(0,1)]#('+','*','of')[random.randint(0,2)]
		}
		return randSig

	def channelSig(self):
		"""
			generates a randome signature for a channel
		"""
		channelSig = {
			'signature':[],
		}
		#loop over and define the wave/soid signature for each channel
		for k in range(0,self.numSoids):
			soidSig = self.randWaveSig()
			channelSig['signature'].append(soidSig)

		return channelSig

	def packetSig(self):
		"""
			generates a random signature for a packet
		"""
		packetSig = {
			'channelSigmas':[],
		}
		#loop over and define the channels for each packet
		for j in range(0,self.numChannels):
			channelSig = self.channelSig()
				
			packetSig['channelSigmas'].append(channelSig)
		return packetSig

	def channelNoise(self):
		"""
			generates some random noies to add to a channel
		"""
		t = self.timesteps
		noise = self.epsilon*torch.randn(t)
		return noise

	def imposition(self):
		"""
			generate an interfering/imposing wave
		"""
		imposition = torch.zeros(self.timesteps)
		if (self.impose and random.randint(0,1)) or self.always_impose:
			for _ in range(0,self.impose):
				imposingWave = self.sgtToWave(self.channelSig())
				imposition += imposingWave

		return imposition

	def sgtToWave(self,sigma):
		"""
			actually generates the data using the signatures
			inputs:
					sigma --[object] frame def:
						{
							timesteps:[int],
							range:(start[float],end[float])
							'signature':[{
								amp:[float],
								freq:[float],
								phase:[float]
							}]
						}
		"""
		t = self.timesteps
		phaseShift = random.uniform(-6.3,6.3)#6.3 ~ 2*pi this needs to be roughly 2*pi*( 1/min(freq) )
		start = self.range[0] + phaseShift
		end = self.range[1] + phaseShift
		linspace = torch.linspace(start,end,t)
		signature = sigma['signature']
		N = len(signature)
		wave = torch.zeros(t)
		for i in range(0,N):
			si = signature[i]
			op = si['op']
			freq = si['freq']
			amp = si['amp']
			phase = si['phase']
			if i == 0:
				wave = self.soid(linspace,amp,freq,phase)
			elif op is '+':
				wave += self.soid(linspace,amp,freq,phase)
			elif op is '*':
				wave *= self.soid(linspace,amp,freq,phase)
			else:
				wave = self.soid(wave,amp,freq,phase)

		return wave

	def soid(self,over,amp,freq,phase):
		"""
			make sinusoidal data using the prams
		"""
		return (amp*torch.sin(over*freq+phase))

	def kill(self,wave):
		if self.dead:
			dead = torch.randn(self.timesteps)+self.dead > 0
			wave *= dead.float()
		return wave

	def subSample(self,wave):
		"""
			extract a sub range from wave 
			and set anything outsid the range to 0
		"""
		t = self.timesteps
		if self.sub_sample:
			minWidth = int(t*self.sub_sample)# roughly ten percent of t

			maxStart = t - minWidth
			start = rint(0,maxStart)

			if(start+minWidth>t):
				minEnd = t

			minEnd = min(start+minWidth,t)
			end = rint(minEnd,t)

			subSample = torch.zeros(t)
			subSample[start:end] = wave[start:end]
			wave = subSample

		return wave

	def scramble(self,wave):
		"""
			apply noise to wave
		"""
		wave = self.subSample(wave)
		wave += self.channelNoise()
		wave += self.imposition()
		wave = self.kill(wave)
		return wave
	
	def packetSample(self,sigma):
		#generates a single sample of a packet

		#sigmas for each channel
		channelSigmas = sigma['channelSigmas']
		#number of channels in the packet
		C = self.numChannels

		t = self.timesteps

		packetInstance = torch.zeros(C,t)
		for i in range(0,C):
			cSi = channelSigmas[i]
			wave = self.sgtToWave(cSi)
			wave = self.scramble(wave)
			
			packetInstance[i,:] = wave
		return packetInstance

	def miniBatch(self,size):
		S = self.Sigma;
		N = self.numPackets
		miniBatchX = torch.zeros(N*size,self.timesteps,self.numChannels)
		miniBatchY = torch.zeros(N*size)
		idx = 0
		for s in range(0,size):
			for i in range(0,N):
				Si = S[i]
				localPacket = self.packetSample(Si)
				lPs = localPacket.shape
				localPacket = localPacket.reshape(lPs[1],lPs[0])
				miniBatchX[idx,:,:] = localPacket
				miniBatchY[idx] = i
				idx+=1

		#permuate the mini-batch to prevent the model from learning
		#the underlying generation pattern
		prm = torch.randperm(N*size)
		miniBatchX = miniBatchX[prm]
		miniBatchY = miniBatchY[prm]

		return (miniBatchX,miniBatchY.long())
