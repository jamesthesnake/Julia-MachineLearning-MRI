#useful example of Knet and Imaging packages, good basis.
using Pkg; for p in ("Colors","ImageMagick","Images"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using Knet, KnetLayers, Colors, ImageMagick, Images, Statistics
import Knet: Data
import KnetLayers: arrtype
setoptim!(model,optimizer) = for p in params(model) p.opt=Knet.clone(optimizer) end # Easy optimizer setter
#Data
include(Knet.dir("data","mnist.jl"))
global const 𝜀=Float32(1e-8)
𝑱d(D,x,Gz) = -mean(log.(D(x) .+ 𝜀) .+ log.((1+𝜀) .- D(Gz)))/2   #discriminator loss
𝑱g(D,G,z)  = -mean(log.(D(G(z)) .+ 𝜀)) # generator loss
𝒩(input, batch) = arrtype(randn(Float32, input, batch))  #sample noise
function runmodel(D, G, data, 𝞗; dtst=nothing, train=false, saveinterval=20)
    gloss = dloss = total= 0.0; B = 𝞗[:batchsize]
    if train
        Dprms, Gprms, L = params(D), params(G), 𝞗[:epochs]
    else
        Dprms, Gprms, L = nothing, nothing, 1
    end

    for i=1:L
        for (x,_) in data
            Gz = G(𝒩(𝞗[:ginp], B)) #Generate Fake Images
            z = 𝒩(𝞗[:ginp], 2B)     #Sample z from Noise

            if train
                jd = @diff 𝑱d(D, x, Gz)
                for w in Dprms update!(w,grad(jd,w))  end
                jg = @diff 𝑱g(D, G, z)
                for w in Gprms update!(w,grad(jg,w))  end
            else
                jd = 𝑱d(D, x, Gz)
                jg = 𝑱g(D, G, z)
            end
            dloss += 2B*value(jd); gloss += 2B*value(jg); total += 2B
        end
        train ? runmodel(D, G, dtst, 𝞗; train=false) : println((gloss/total, dloss/total))
        i % saveinterval == 0 && generate_and_show(D, G, 100, 𝞗)  # save 10 images
    end
end
function generate_and_show(D,G,number,𝞗)
    Gz    = convert(Array,G(𝒩(𝞗[:ginp], number))) .> 0.5
    Gz    = reshape(Gz, (28, 28, number))
    L     = floor(Int, sqrt(number))
    grid  = []
    for i = 1:L:number
        push!(grid, reshape(permutedims(Gz[:,:,i:i+L-1], (2,3,1)), (L*28,28)))
    end
    display(Gray.(hcat(grid...)))
end


𝞗 = Dict(:batchsize=>32,:epochs=>80,:ginp=>256,:genh=>512,:disch=>512,:optim=>Adam(;lr=0.0002))
Go = Chain(MLP(𝞗[:ginp], 𝞗[:genh], 784; activation=ELU()), Sigm())
Ds = Chain(MLP(784, 𝞗[:disch], 1; activation=ELU()), Sigm())
setoptim!(Ds, 𝞗[:optim]); setoptim!(Go, 𝞗[:optim])

xtrn,ytrn,xtst,ytst = mnist()
global dtrn,dtst = mnistdata(xsize=(784,:),xtype=arrtype, batchsize=𝞗[:batchsize])
generate_and_show(Ds, Go, 100, 𝞗)
runmodel(Ds, Go, dtst, 𝞗; train=false) # initial losses
runmodel(Ds, Go, dtrn, 𝞗; train=true, dtst=dtst) # training
