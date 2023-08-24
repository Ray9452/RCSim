using Distributed
@everywhere include("FixedStart.jl")
using DataFrames, XLSX, Random, Statistics, LinearAlgebra, SymPy
function GetSides(periodic,max_radius,number_of_particles)::Float64
    if ! periodic
        # Define the symbols
        @vars L rm Ns ϕ
        # Define the equation
        eq = Eq(Ns*π*rm^2/L^2, ϕ - ((L^2 - (L-rm)^2)/L^2)*(ϕ - (π/4)))
        # Substitute values for the variables
        values = Dict(rm => max_radius, Ns => number_of_particles, ϕ => 0.84)
        eq_substituted = eq.subs(values)
        # Solve the equation for L
        solutions = solve(eq_substituted, L)
        # Filter for real, positive solutions
        real_positive_solutions = [sol for sol in solutions if isreal(sol) && sol > 0]
        #sim parameters to think about
        side = real_positive_solutions[1]
    else
        side=sqrt((number_of_particles*π*max_radius^2)/0.84)
    end
    return side
end
function off_lattice_section_sim(parm; printy=false)
    # this produces data that plots ending fraction space as a function of killing ratio given some space size
    num_cells = parm["num_cells"]
    big_kill = parm["big_kill"]
    number_of_partitions = parm["number_of_partitions"]
    date = parm["index"]
    ratio_blue = parm["ratio_blue"]
    gen = parm["gen"]

    # RED IS BTR killer
    # Data setup
    columns = ["inf_space", "ratio_speed", "standard_deviation", "num_sections", "num_cells", "date", "generations","ratio of inferior","big_kill","periodic","relax"]
    ind = range(big_kill, stop=0, length=20)
    df = DataFrame(; (Symbol(c) => Vector{Union{Missing, Any}}(missing, length(ind)) for c in columns)...)

    df[1, "num_sections"] = number_of_partitions
    df[1, "num_cells"] = num_cells
    df[1, "ratio of inferior"] = ratio_blue
    df[1, "generations"] = gen
    df[1, "date"] = date
    df[1, "big_kill"] = big_kill
    df[:, "ratio_speed"] .= ind ./ big_kill
    df[1,"periodic"] = parm["periodic"]
    df[1,"relax"] = parm["relaxation"]
    filename = "Partition_killing_$date"
    li = length(ind)
    ni = 1
    side=GetSides(parm["periodic"],parm["max_radius"],num_cells)
    sides::SVector{2,Float64}=SVector{2,Float64}([side, side])
    
    for i in eachindex(ind)
        blue_nums = rand(Binomial(num_cells, ratio_blue), number_of_partitions)
        blue_kill = ind[i]
        rF = zeros(number_of_partitions)
        bF = zeros(number_of_partitions)
        tot = zeros(number_of_partitions)
        devi = zeros(number_of_partitions)
        for x in 1:number_of_partitions
            blue_start=blue_nums[x]/num_cells
            model=initialize_model(
                number_of_particles=num_cells,
                carrying_capacity=num_cells,
                sides=sides,
                dt=parm["dt"],
                min_radius=parm["min_radius"],
                max_radius=parm["max_radius"], #so as to double the area when this is reached
                parallel=true,
                spring_constant=parm["spring_constant"],
                s_i=blue_start,
                k_r=blue_kill,
                periodic=parm["periodic"],
                wall_constant=parm["wall_constant"])
            try
                simulate(model; nsteps=gen*100, early_stop=true, save_every=50)
            catch e
                println("error")
                simulate(model; nsteps=gen*100, early_stop=true, save_every=50)
                #give it one more shot
            end
            
            red_final::Integer = model.reds 
            blue_final::Integer = model.blues
            rF[x] = red_final
            bF[x] = blue_final
            tot[x] = red_final + blue_final
        end
        inf_space = round(sum(bF) / sum(tot), digits=3)
        df[i, "inf_space"] = inf_space
        devi .= bF ./ tot
        df[i, "standard_deviation"] = std(devi)
        if printy
            println("$filename $(ni/li) done")
        end
        ni += 1
    end

    done = false
    iter = 1
    while !done
        if isfile(joinpath(pwd(), "$filename.xlsx"))
            if iter == 1
                filename = filename * "(1)"
            else
                filename = filename[1:end-3] * "($iter)"
            end
            iter += 1
        else
            XLSX.writetable("$filename.xlsx", collect(DataFrames.eachcol(df)), DataFrames.names(df), sheetname="frac_corelate")
            done = true
            if printy
                println("Done with $filename")
            end
        end
    end
end
function main()
    #sim parameters to not think about
    dt::Float64=0.01 #in generations 
    min_radius::Float64=0.5
    max_radius::Float64=min_radius*sqrt(2)
    spring_constant::Int64=500
    wall_constant::Int64=1000

    #to think about
    s_i::Float64 = 0.5
    nums=[5,10,100,1000]
    bounds=[true,false]
    points=Dict(
        5=>1000,
        10=>1000,
        100=>100,
        1000=>10
    )
    i=1
    for num in nums; bound in bounds; relax in bounds
        parm = Dict(
            "num_cells" => num,                  
            "big_kill" => 1.0,                   
            "number_of_partitions" => points[num],        
            "index" => String(i),    
            "ratio_blue" => s_i,        
            "gen" => 64,         
            "dt" => dt,        
            "min_radius" => min_radius,         
            "max_radius" => max_radius,         
            "spring_constant" => spring_constant,         
            "periodic" => bound,        
            "wall_constant" => wall_constant,
            "relaxation" => relax          
        )
        off_lattice_section_sim(parm)
        i+=1
    end
end
#main()

#sim parameters to not think about
dt::Float64=0.01 #in generations 
parallel::Bool = true
min_radius::Float64=0.5
max_radius::Float64=min_radius*sqrt(2)
spring_constant::Int64=500
periodic=false
wall_constant::Int64=1000

#to think about
number_of_particles::Int64=1025
carrying_capacity::Int64=number_of_particles
s_i::Float64 = 0.5
k_r::Float64 = 0.6
side=GetSides(periodic,max_radius,number_of_particles)
sides::SVector{2,Float64}=SVector{2,Float64}([side, side])
model=initialize_model(
    number_of_particles=number_of_particles,
    carrying_capacity=carrying_capacity,
    sides=sides,
    dt=dt,
    min_radius=min_radius,
    max_radius=max_radius, #so as to double the area when this is reached
    parallel=parallel,
    spring_constant=spring_constant,
    s_i=s_i,
    k_r=k_r,
    periodic=periodic,
    wall_constant=wall_constant)
simulate(model; relaxation=false,nsteps=10,early_stop=false,save_every=1)
model=initialize_model(
    number_of_particles=number_of_particles,
    carrying_capacity=carrying_capacity,
    sides=sides,
    dt=dt,
    min_radius=min_radius,
    max_radius=max_radius, #so as to double the area when this is reached
    parallel=parallel,
    spring_constant=spring_constant,
    s_i=s_i,
    k_r=k_r,
    periodic=periodic,
    wall_constant=wall_constant)
#@time simulate(model; nsteps=steps)
@profilehtml simulate(model; relaxation=false, nsteps=6400,early_stop=false,save_every=6400)
@time agent_data=simulate(model; relaxation=true,nsteps=6400,early_stop=false,save_every=6400)
#base_graphs(agent_data)
anim = animation_better(agent_data,model.sides)
gif(anim,"Test.gif")

# Define the parm dictionary with placeholder values
# Initialize the parm dictionary with placeholder values



# Define when to save data
#save_every = 1
#whensave = Array{Int64}(1:save_every:steps)
#@time agent_data, model_data = run!(model, agent_step!, model_step!, steps; when=whensave,adata=adata)
#CSV.write("Test.csv",agent_data)
#base_graphs(agent_data)
#q[!,:color]=Symbol.(q[!,:color])