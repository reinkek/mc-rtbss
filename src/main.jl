using Distributions: MvNormal, pdf, logpdf
using PyPlot: plot, axes, xlim, ylim, pause, cla
include("util.jl")

#action = Vector{Float64}[[1,0], [0,1], [-1,0], [0,-1], [0,0], [0,0]]
#action = Vector{Float64}[[0,1], [0,-1], [0,0], [0,0]]
#action = action*0
action = Vector{Float64}[[0,0], [0,0]]
numActions = length(action)

SHOOT = numActions
WAIT = SHOOT-1
numStep = 2
dt = 0.1
numParticles = 200
D = 2
sensorCov = diagm([1;100])
sensorCovInv = diagm([1;1/100])
normalizer = 1/2/pi/sqrt(det(sensorCov))

startState = trajectoryNominal(0.0)
stateTrue = reshape(startState[1:2],2,1)
xAgent = zeros(2)
b = 30*(rand(2,numParticles) - 0.5)
#b = [[10.0,0.0] [0.0,10.0]];

axes(aspect=1)
xlim([-15,15])
ylim([-15,15])
plot(b[1,:],b[2,:],"co")
plot(xAgent[1],xAgent[2],"ro")
plot(stateTrue[1],stateTrue[2],"mo")

obsMod, sampMod = getObservationModel(xAgent, stateTrue[1:2])
obsSample = vec(sampMod(1))
plot(obsSample[1],obsSample[2],"go")

pause(0.5)

b = updateBeliefState(b, xAgent, obsSample)

cla()
xlim([-15,15])
ylim([-15,15])
plot(b[1,:],b[2,:],"bo")
plot(xAgent[1],xAgent[2],"ro")
plot(stateTrue[1],stateTrue[2],"mo")
plot(obsSample[1],obsSample[2],"go")
pause(.5)

t = 0.0
while(t<10)
    v, a = expandTree(b, xAgent, D, t)

    println(a,xAgent,action[a])
    stateTrue = posStateTransition(stateTrue,t)
    b = posStateTransition(b, t)
    xAgent += action[a]

    cla()
    xlim([-15,15])
    ylim([-15,15])
    plot(b[1,:],b[2,:],"co")
    plot(xAgent[1],xAgent[2],"ro")
    plot(stateTrue[1],stateTrue[2],"mo")
    pause(0.5)

    obsMod, sampMod = getObservationModel(xAgent, stateTrue[1:2])
    obsSample = vec(sampMod(1))
    b = updateBeliefState(b, xAgent, obsSample)

    cla()
    xlim([-15,15])
    ylim([-15,15])
    plot(b[1,:],b[2,:],"bo")
    plot(xAgent[1],xAgent[2],"ro")
    plot(stateTrue[1],stateTrue[2],"mo")
    plot(obsSample[1],obsSample[2],"go")
    pause(.5)

    t += dt*numStep
end



readline(STDIN)
