gam = 0.999
nObs = 10
numSort = 10

function updateBeliefState(b::Matrix{Float64},xAgent::Vector{Float64},obs::Vector{Float64})
    #println("updateBeliefState")
    M = size(b)[2]
    scores = zeros(M)
    for i = 1:M
      #(obsMod,obsSampler) = getObservationModel(xAgent,b[:,i])
      #scores[i] = obsMod(obs)
      scores[i] = getObservationModelDensity(xAgent,b[:,i],obs)
      #println(scores[i]," , ",scoreNew)
    end
    bp = zeros(Float64,size(b))
    avgScore = sum(scores)/M
    r = rand()*avgScore
    c = scores[1]
    i = 1
    for m = 1:M
        u = r + (m-1)*avgScore
        while(u>c)
            i += 1
            c += scores[i]
        end
        bp[:,m] = b[:,i]
    end
    bp
end

function rewardModel(x::Vector{Float64},b::Matrix{Float64},a::Int)
    #println("rewardModel")
    timeCost = -100
    shotCost = 0
    hitReward = 1000
    moveCost = -100
    r = timeCost
    if(a == SHOOT)
        r += shotCost
        angleToTgt = atan2(b[2,:]-x[2],b[1,:]-x[1])
        binEdge = -pi:(pi/180):pi
        
        (binEdge,counts) = hist(angleToTgt[:],binEdge)
        
        (maxCount,maxIdx) = findmax(counts)
        r += (1-(maxCount/size(b)[2]))*timeCost/(1-gam)#hitReward*(maxCount/size(b)[1])
        #r += hitReward*(maxCount/size(b)[2])
    elseif(a < WAIT)
        r += moveCost
    end
    r
end

function getObservationModel(xAgent::Vector{Float64}, obs::Vector{Float64})
    #println("getObsModel")
    #println(obs)
    #println(xAgent)
    ang = atan2((obs-xAgent)[2],(obs-xAgent)[1])
    R = [cos(ang) -sin(ang) ; sin(ang) cos(ang)]
    sig = R*sensorCov*R'
    #println(sig)
    d = MvNormal(obs,sig)
    densityFunction = f -> pdf(d,f)
    samplingFunction = r -> rand(d,r)
    (densityFunction,samplingFunction)
end

function getObservationModelDensity(xAgent::Vector{Float64},obs::Vector{Float64}, x::Vector{Float64})
    ang = atan2(obs[2]-xAgent[2],obs[1]-xAgent[1])
    R = [cos(ang) -sin(ang) ; sin(ang) cos(ang)]
    sigInv = R*sensorCovInv*R'
    #rho = sig[1,2]/(sig[1,1]*sig[2,2])
    err = x-obs
    #z = (err[1]/sig[1,1])^2 - 2*rho*err[1]*err[2]/(sig[1,1]*sig[2,2]) + (err[2]/sig[2,2])^2
    #exp( -z/(2*(1-rho)^2) )
    exp(-1/2*err'*sigInv*err)[1]
end

function getObservationModelSample(xAgent::Vector{Float64},obs::Vector{Float64})
    ang = atan2(obs[2]-xAgent[2],obs[1]-xAgent[1])
    R = [cos(ang) -sin(ang) ; sin(ang) cos(ang)]
    sig = R*sensorCov*R'
    rho = sig[1,2]/(sig[1,1]*sig[2,2])
    sig*randn(2)+obs
end

function fullStateTransition(stateTarget, t)
    #println("fullStateTrans")
    p = 0.02*randn(2,numStep)
    k = [2*randn(numStep)+5  4*randn(numStep)+10]
    b = [1.5*randn(numStep)+2   2*randn(numStep)+2]
    stateNext = stateTarget

    for i in 1:numStep
        stateNext[1:2] += dt * stateNext[3:4] + p[:,i]
        A = dt * [k[i,1]  0     b[i,1]   0
                  0    k[i,2]  0     b[i,2]]
        stateNext[3:4] += A*(trajectoryNominal(t) - stateNext)
    end

    return stateNext
end

## returns the new particles after a transition period, given a current set of particles
function posStateTransition(posTarget, t)
    #println("posStateTrans")
    numSamples = size(posTarget)[2]    
    stateNominal = trajectoryNominal(t)
    stateNext = vcat(posTarget,0.5*randn(2,numSamples) .+ stateNominal[3:4])
    for i in 1:numStep
        p = 0.02*randn(2,numSamples)
        k = [2*randn(numSamples)+20  5*randn(numSamples)+25]'
        b = [3*randn(numSamples)+1   2*randn(numSamples)+2]'
        ## dot-minus for singleton expansion
        err = trajectoryNominal(t) .- stateNext

        stateNext[1:2,:] += dt * stateNext[3:4,:] + p
        stateNext[3,:] += dt*(k[1,:].*err[1,:] + b[1,:].*err[3,:])
        stateNext[4,:] += dt*(k[2,:].*err[2,:] + b[2,:].*err[4,:])
    end

    return stateNext[1:2,:]
end

function particleProject(sampledStatesB::Matrix{Float64}, xAgentNew::Vector{Float64}, t::Float64, a::Int)
    #println("particleProject")
    numSamplesB = size(sampledStatesB)[2]
    
    sampledStatesBP = posStateTransition(sampledStatesB, t)
    
    iSamplesO = rand(1:numSamplesB, nObs)
    sampledStatesObsP = posStateTransition(sampledStatesB[:,iSamplesO], t)
    
    Bp = {}
    for i in 1:nObs
        obsSampler = getObservationModel(xAgentNew,sampledStatesObsP[:,i])[2]
        sampledObs = obsSampler(1)[:,1]
        #sampledObs = getObservationModelSample(xAgentNew,sampledStatesObsP[:,i])
        #println(size(sampledObs))
        push!(Bp,updateBeliefState(sampledStatesBP, xAgentNew,sampledObs)) 
    end
    
    return Bp
end

function expandTree(b::Matrix{Float64}, xAgent::Vector{Float64}, depth::Int, t::Float64)
    aStar = 0
    if depth == 0
        return LowerBound(b,xAgent)
    else
        pq = Collections.PriorityQueue{(Int,Float64),Float64}(Base.Order.Reverse)
        for a in 1:numActions
            ub = UpperBound(b,xAgent,t,a)
            pq[(a,ub)] = ub
        end
        lowerT = -Inf
        (a,ub) = Collections.dequeue!(pq)
        while ub > lowerT
            #println(a,' ',ub,' ',depth)
            xAgentNew = xAgent + action[a]
            B = particleProject(b, xAgentNew, t, a)
            sum = 0
            if(a!=SHOOT)
                for bp in B
                    sum += expandTree(bp, xAgentNew, depth-1,t)[1]
                end
            end
            lowerTa = rewardModel(xAgent, b, a) + gam/nObs*sum
            if lowerTa > lowerT
                aStar = a
                lowerT = lowerTa
            end
            if(!isempty(pq))
                (a,ub) = Collections.dequeue!(pq)
            else
                break;
            end
        end
    end
    return (lowerT,aStar)
end

function LowerBound(b::Matrix{Float64}, xAgent::Vector{Float64})
    #println("LowerBound")
    return rewardModel(xAgent, b, SHOOT)
end


function UpperBound(b::Matrix{Float64}, xAgent::Vector{Float64}, t::Float64, a::Int)
    #println("UpperBound")
    
    numSamplesB = size(b)[2]
    iSamples = rand(1:numSamplesB,numSort)
    sampledStates = posStateTransition(b[:,iSamples],t)
    return rewardModel(xAgent,sampledStates,a)
end

function trajectoryNominal(t::Float64)
    #println("trajNominal")
    return [10.0
            0.0
            0.0
            0.0]
    #return [5*cos(pi*t/20)+2
    #        10*sin(pi*t/20)+1
    #        -pi/4*sin(pi*t/20)
    #        pi/2*cos(pi*t/20)]
end
