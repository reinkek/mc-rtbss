using Distributions: MvNormal, pdf, logpdf
#using PyPlot: plot, axes, xlim, ylim, pause, cla
include("util.jl")

#action = Vector{Float64}[[1,0], [0,1], [-1,0], [0,-1], [0,0], [0,0]]
action = Vector{Float64}[[0,0], [0,0]]
numActions = length(action)

SHOOT = numActions
WAIT = SHOOT-1
numStep = 2
dt = 0.1
numParticles = 200
D = 2
sensorCov = diagm([0.1;1000])
sensorCovInv = diagm([10;1/1000])
normalizer = 1/2/pi/sqrt(det(sensorCov))

startState = trajectoryNominal(0.0)
stateTrue = reshape(startState[1:2],2,1)
xAgent = zeros(2)
b = 30*(rand(2,numParticles) - 0.5)

obsMod, sampMod = getObservationModel(xAgent, stateTrue[1:2])
obsSample = vec(sampMod(1))
b = updateBeliefState(b, xAgent, obsSample)
expandTree(b, xAgent, 0, 0.0)

@time for i = 1:10000; (f,g) = getObservationModel(xAgent,stateTrue[1:2]); f(xAgent); end

@time for i = 1:10000; getObservationModelDensity(xAgent,stateTrue[1:2],xAgent); end

@time for i = 1:1000; updateBeliefState(b,xAgent,obsSample); end 

@time for i = 1:30; expandTree(b,xAgent,1,0.0); end