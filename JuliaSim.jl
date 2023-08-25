using CellListMap.PeriodicSystems, StaticArrays, Random, Agents, Distributions, Plots, StatsPlots, DataFrames, StatProfilerHTML, CSV,  GeometryBasics, SharedArrays, Debugger
using Base.Iterators: accumulate
#the ContinuousAgent needs a velocity, but as these cells are sticky I just set all calls to zero
@agent Particle ContinuousAgent{2} begin
    r::AbstractFloat # radius
    k::AbstractFloat # repulsion force constant
    mass::AbstractFloat
    growth_rate::AbstractFloat
    kill_rate::AbstractFloat
    color::Symbol
    IsPresent::Bool
end
MakeParticle(; id, pos, r, k, mass, growth_rate,kill_rate,color,isPresent) = Particle(id, pos, (0.0,0.0), r, k, mass, growth_rate,kill_rate,color,isPresent) #the (0,0) is velocity
Base.@kwdef mutable struct Results
    current_mass::Vector{Float64}
    new_mass::Vector{AbstractFloat}
    current_r::Vector{AbstractFloat}
    new_r::Vector{AbstractFloat}
    kill_list::Vector{Integer}
    divide_list::Vector{Integer}
    activation_order::Vector{Integer}
end
Base.@kwdef mutable struct Parameters
    dt::AbstractFloat
    number_of_particles::Integer
    sides::SVector{2,<: Number}
    max_radius::AbstractFloat 
    min_radius::AbstractFloat 
    max_mass::AbstractFloat
    min_mass::AbstractFloat
    avaliable_ids::Vector{<: Integer}
    system::PeriodicSystems.PeriodicSystem1
    s_i::AbstractFloat
    k_r::AbstractFloat
    step::Integer
    area::AbstractFloat
    periodic::Bool
    wall_constant::Integer
    carrying_capacity::Integer
    reds::Integer
    blues::Integer
    relaxation::Bool
    results::Results
    packing_fraction::AbstractFloat
    max_packing_fraction::AbstractFloat
    pf_limited::Bool 
    cc_limited::Bool
end
Base.@kwdef mutable struct Outputs
    forces::Vector{SVector{2,<: AbstractFloat}}
    kill_list::Vector{Vector{Integer}}
end
import CellListMap.PeriodicSystems: copy_output, reset_output!, reducer
copy_output(x::Outputs) = Outputs(deepcopy(x.forces), deepcopy(x.kill_list))
function reset_output!(x::Outputs)
    fill!(x.forces, zero(x.forces[begin]))
    fill!(x.kill_list,Vector{Integer}())
    return x
end
function reducer(x::Outputs, y::Outputs)
    combined::Vector{Vector{Integer}} = vcat.(x.kill_list,y.kill_list)
    x.forces .+= y.forces
    return Outputs(x.forces,combined)
end
function initialize_model(;
    number_of_particles::Integer,
    carrying_capacity::Integer,
    sides::SVector{2,<: Number},
    dt::AbstractFloat,
    min_radius::AbstractFloat,
    max_radius::AbstractFloat, #so as to double the area when this is reached
    parallel::Bool,
    spring_constant::Integer,
    s_i::AbstractFloat,
    k_r::AbstractFloat,
    periodic::Bool,
    wall_constant::Integer,
    max_packing_fraction::AbstractFloat,
    pf_limited::Bool,
    cc_limited::Bool)::ABM

    max_mass::AbstractFloat=π*(max_radius^2)
    min_mass::AbstractFloat=π*(min_radius^2)
    # initial random positions for cells 
    positions::Vector{SVector{2,Float64}} = [sides .* rand(SVector{2,Float64}) for _ in 1:carrying_capacity]
    # We will use CellListMap to compute forces, with similar structure as the positions
    forces::Vector{SVector{2,<: AbstractFloat}} = similar(positions)
    #kill list start 
    kill_list::Vector{Vector{Integer}} = fill(Vector{Integer}(), carrying_capacity)
    # Space and agents
    # Initialize CellListMap periodic system which will be an attribute of the ABM 'model'
    system = PeriodicSystem(
        positions=positions,
        unitcell=sides,
        cutoff=2 * max_radius,
        output=Outputs(forces,kill_list),
        output_name=:outputs, # allows the system.forces alias for clarity
        parallel=parallel,
    )
    #have avaliable_ids ready to go
    avaliable_ids::Vector{<: Integer} = Vector{Integer}(number_of_particles+1:carrying_capacity)
    # define the model properties for the ABM 
    # The clmap_system field contains the data required for CellListMap.jl
    step::Integer=1
    area::AbstractFloat = sides[1]*sides[2]
    #the agents will have a raidus and we get them here
    radiuss::Vector{Float64}=rand(Uniform(min_radius,max_radius),number_of_particles)
    append!(radiuss,zeros(carrying_capacity-number_of_particles))
    growth_rates::Vector{<: AbstractFloat}=rand(Uniform(0.9,1.1),carrying_capacity).*(π*(min_radius^2))
    masss::Vector{<: AbstractFloat}=(radiuss.^2)*π
    packing_fraction=sum(masss)/area
    blues::Integer = round(Integer,s_i*number_of_particles)
    blues=blues 
    reds=number_of_particles-blues
    properties = Parameters(
        dt,
        number_of_particles,
        sides,
        max_radius,
        min_radius,
        max_mass,
        min_mass,
        avaliable_ids,
        system,
        s_i,
        k_r,
        step,
        area,
        periodic,
        wall_constant,
        carrying_capacity,
        reds,
        blues,
        false,
        Results(masss,masss,radiuss,radiuss,Int[],Int[],Int[]),
        packing_fraction,
        max_packing_fraction,
        pf_limited,
        cc_limited)
    #Recall from above we have defined what the Particle struct is 
    scheduler = Schedulers.randomly #make agents run in a random order
    model = AgentBasedModel(Particle,
        properties=properties,
        scheduler=scheduler
    )
    # Create active agents with random radius
    for id in 1:carrying_capacity
        radius::AbstractFloat=radiuss[id]
        cur_mass::AbstractFloat=masss[id]
        cur_gr::AbstractFloat=growth_rates[id]
        color = (id<=blues) ? :blue : :red
        cur_k =  color==:blue ? k_r : 1
        cur_pos=Tuple(positions[id])
        isPresent=true
        if id>number_of_particles
            isPresent=false 
        end
        add_agent!(
            MakeParticle(
                id=id,
                r=radius,
                k=spring_constant, # force constants
                mass=cur_mass, # random masses
                pos=cur_pos,
                growth_rate=cur_gr,
                kill_rate=cur_k, #as the faster kills with rate =1 the slower kills with just k_r
                color = color,
                isPresent=isPresent 
            ),model)
    end
    return model #model is type ABM with an attribute that is CellListMap
end
function model_step!(model::ABM)
    # Update the pairwise forces at this step in the .system attribute
    map_pairwise!(
        (x, y, i, j, d2, forces) -> calc_forces!(x, y, i, j, d2, forces, model),
        model.system,
    )
    agents_parallel!(model) #makes all the cells move, and gets the results and changes the .field in place
    if ! model.relaxation
        kill_from_results!(model) #death can always happen
        filter_results_from_death!(model) #remove killed cells from mass/r/activation
        model.packing_fraction = (sum(model.results.current_mass))/model.area #as any dead cells during the step should have 0 mass 
        grow_model!(model)
        divide_model!(model)
        model.step+=1
    end
    return nothing
end
function filter_results_from_death!(model::ABM)
    #we want a clean decision as to who grows and divides so we need to change so change the mass/radius in results and remove killed cell ids from the divide list
    # Convert kill_list to a Set for O(1) lookups
    kill_set = Set(model.results.kill_list)
    for i in eachindex(model.results.activation_order)
        if (model.results.activation_order[i] in kill_set) #just died (kill set will always be small so this shouldnt be too long)
            model.results.current_mass[i]=0
            model.results.new_mass[i]=0
            model.results.current_r[i]=0
            model.results.new_r[i]=0
        end
    end
    filter!(x -> !(x in kill_set),model.results.divide_list) #not in the divide list now 
end
function divide_model!(model::ABM)
    num_possible=length(model.results.divide_list)
    done=false
    i=1
    ids=model.results.divide_list
    if (length(ids)==0) || ((model.number_of_particles>=model.carrying_capacity)) || (i > num_possible)
        return nothing 
    end
    while ! done
        @inbounds cur_id=ids[i]
        agent_divide!(model[cur_id],model)
        i+=1
        if ((model.number_of_particles>=model.carrying_capacity)) || (i > num_possible)
            done=true
        end
    end
end
function grow_model!(model::ABM)
    if model.pf_limited #only some grow in this case
        if model.packing_fraction >= model.max_packing_fraction
            return nothing
        end
        change_in_mass::Vector{Float64} = (model.results.new_mass -  model.results.current_mass)
        idx=idx_of_cumulative_value(change_in_mass,(model.max_packing_fraction-model.packing_fraction)*model.area) #we update the packing fraction at each time step 
        if idx===nothing
            return nothing 
        end
        model.packing_fraction=(sum(change_in_mass[1:idx]) + model.packing_fraction*model.area)/model.area
        #only put in ids with living cells to save on headaches 
        ids = Int[]
        new_r = Float64[]
        new_mass = Float64[]
        for cur_id in model.results.activation_order[1:idx]
            if (model.results.current_mass[cur_id] != 0 )
                push!(ids,cur_id)
                push!(new_r,model.results.new_r[cur_id])
                push!(new_mass,model.results.new_mass[cur_id])
            end
        end
    elseif model.cc_limited #everyone grows 
        ids=model.results.activation_order
        new_r=model.results.new_r
        new_mass=model.results.new_mass
    end
    @sync for i in eachindex(ids)
        Threads.@spawn _grow_parallel(ids[i],new_r[i],new_mass[i]) 
    end
end
function _grow_parallel(id::Integer,new_r::AbstractFloat,new_mass::AbstractFloat)
    agent=model[id]
    agent.r=new_r
    agent.mass=new_mass
end
function calc_forces!(x::SVector{2,<: AbstractFloat}, y::SVector{2,<: AbstractFloat}, i::Integer, 
    j::Integer, d2::AbstractFloat, outputs::Outputs, model::ABM)
    @inbounds pᵢ::Particle = model[i]
    @inbounds pⱼ::Particle = model[j]
    if (! pᵢ.IsPresent) || (! pⱼ.IsPresent)
        return outputs
    end
    d = sqrt(d2)
    overlap = (pᵢ.r + pⱼ.r) - d
    if overlap > 0  
        dr = @. y - x
        fij = @. pᵢ.k * overlap * (dr/d)
        @inbounds outputs.forces[i] -= fij #as the vector points to the second, we want to subtract it from the first
        @inbounds outputs.forces[j] += fij
        #now as we know they overlap, we potentially add them as atackers to each other 
        if (pᵢ.color != pⱼ.color) && (! model.relaxation)
            @inbounds push!(outputs.kill_list[i],j)
            @inbounds push!(outputs.kill_list[j],i)
        end
    end
    return outputs
end
function agents_parallel!(model::ABM)
    # In parallel, we can have all the cells move/grow
    # Collect where new cells must be added/removed and then we do that in series
    activation_order::Vector{Int64} = Agents.schedule(model)
    # This will now store the results of the tasks, not the tasks themselves
    futures = Vector{Vector{Union{Integer, AbstractFloat}}}(undef, length(activation_order))
    @sync for (idx, id) in enumerate(activation_order)
        # Spawn the task with the data and store the result directly in futures
        Threads.@spawn begin
            result = agent_parallel_activate($model.agents[id], $model.max_radius, $model.relaxation, 
                $model.periodic, $model.dt, $model.max_mass, $model.wall_constant, 
                $model.sides, model)
            futures[idx] = result
        end
    end
    # Now, the futures vector contains the results of all tasks
    agent_parallel_reduce!(futures, activation_order)
end    
function agent_parallel_reduce!(futures::Vector{Vector{Union{Integer, AbstractFloat}}},activation_order::Vector{Int64})
    # Initialize result
    mass::Vector{AbstractFloat} = []
    new_mass::Vector{AbstractFloat} = []
    current_r::Vector{AbstractFloat}=[]
    new_r::Vector{AbstractFloat}=[]
    kill_list::Vector{Integer} = []
    divide_list::Vector{Integer} = []
    # Fetch results and reduce
    for fut in futures
        append!(mass, fut[1])
        append!(new_mass,fut[2])
        append!(current_r, fut[3])
        append!(new_r, fut[4])
        append!(kill_list, fut[5])
        append!(divide_list, fut[6])
    end
    filter!(x -> x ≠ 0, kill_list)
    filter!(x -> x ≠ 0, divide_list)
    model.results=Results(mass,new_mass,current_r,new_r,kill_list,divide_list,activation_order)
end
function agent_parallel_activate(agent::Particle,max_radius::AbstractFloat,relaxation::Bool,periodic::Bool,
    dt::AbstractFloat,max_mass::AbstractFloat,wall_constant::Number,sides::SVector{2,<: AbstractFloat},
    model::ABM)
    result::Vector{Union{Integer, AbstractFloat}} = Vector{Union{Integer, AbstractFloat}}()
    push!(result,agent.mass)
    push!(result,agent.mass)
    push!(result,agent.r)
    push!(result,agent.r)
    push!(result,0)
    push!(result,0)
    if ! agent.IsPresent
        return result
    end
    id=agent.id
    #grow radius and mass 
    divide::Bool = agent.r >= max_radius
    if (! divide) && (! relaxation) #no grow if able to divide or relaxing
        #if carrying_capacity is non-zero then you can grow however you want 
        del_r::AbstractFloat=(agent.growth_rate .* dt / (2*π*agent.r))
        new_r::AbstractFloat=del_r .+ agent.r
        #some divide if the number_of_particles is less than the carrying capacity
        divide=new_r > max_radius 
        new_r=divide ? max_radius : new_r # in the event we can divide but wont due to carrying_capacity
        new_mass = divide ? max_mass : π*(new_r^2)
        result[2]=new_mass
        result[4]=new_r #new radius 
    end
    #deal with killing now
    @inbounds kill_list::Vector{Integer}=model.system.outputs.kill_list[id]
    if (! isempty(kill_list)) && (! relaxation)
        prob::AbstractFloat = agent.kill_rate*dt
        if rand()<prob
            looser_index::Integer = rand(1:length(kill_list))
            looser::Integer =  kill_list[looser_index]
            @inbounds result[5]=looser
        end
    end
    if divide
        @inbounds result[6]=id
    end
    cur_pos=SVector{2, <: AbstractFloat}(agent.pos)
    # Retrieve the forces on agent id
    @inbounds f::SVector{2, <: AbstractFloat} = model.system.outputs.forces[id]
    if ! periodic
        wf::SVector{2,AbstractFloat}=wall_force(cur_pos, agent, wall_constant,sides)
        f = @. f + wf
    end
    a = @. f / agent.mass
    # Update positions and velocities
    v = @. a * dt
    new_pos = @. cur_pos.+(v.*dt) 
    if periodic
        new_pos = enforce_periodic_bounds(new_pos,sides)
    end
    agent.pos=Tuple(new_pos)
    # !!! IMPORTANT: Update positions in the CellListMap.PeriodicSystem
    @inbounds model.system.positions[id] = SVector{2,AbstractFloat}(agent.pos)
    return result
end
function kill_from_results!(model::ABM)
    #death can always happen 
    new_kill_list = Int[]
    for looser in model.results.kill_list
        cell=model[looser]
        if cell.IsPresent #just a check just in case
            remove_cell!(cell,model)
            append!(new_kill_list,looser)
        end
    end
    model.results.kill_list=new_kill_list
    return nothing
end
function remove_cell!(cell::Particle, model::ABM)
    push!(model.avaliable_ids,cell.id)
    #update celllistmap by launching the position into oblivion until needed to avoid calc forces with it (lord help me if this works)
    new_pos=SVector{2,AbstractFloat}(cell.pos)+(rand(Uniform(1,10)))*model.sides
    cell.pos=Tuple(new_pos)
    @inbounds model.system.positions[cell.id]=SVector{2,AbstractFloat}(new_pos)
    if cell.color==:blue
        model.blues-=1
    else
        model.reds-=1
    end
    model.number_of_particles -=1 
    #now for the model where we check this parameter and ignore this agent if it is false
    #as the bounds are periodic, it doesnt matter to call the move_agent for the ABM 
    cell.IsPresent=false
    cell.r=0
    cell.mass=0
    #cell.kill_list = Vector{Integer}()
    return nothing
end
function idx_of_cumulative_value(vec::Vector{T}, value::T) where T
    # Compute the cumulative sum
    cumsum_vec = accumulate(+, vec)
    # Convert the Accumulate iterator to an array
    cumsum_array = collect(cumsum_vec)
    # Find the index where the cumulative sum equals or exceeds the set value
    idx = findfirst(>=(value), cumsum_array)
    # If no such index is found, return the entire vector
    if (idx === nothing) && (cumsum_array[end]<value)
        return length(vec)
    elseif idx===1
        return nothing
    end
    # Slice the vector from 1 to the found index
    return idx
end
function enforce_periodic_bounds(position::SVector{2, T}, sides::SVector{2, T}) where T
    new_x = mod(position[1], sides[1])
    new_y = mod(position[2], sides[2])
    return SVector(new_x, new_y)
end
function agent_divide!(agent::Particle,model::ABM)
    if (! agent.IsPresent) || (agent.mass < model.max_mass) || (agent.r < model.max_radius)
        return nothing
    end
    pos1::SVector{2,<: AbstractFloat}=model.system.positions[agent.id]
    θ::AbstractFloat=2*π*rand()
    unit_vector::SVector{2,<: AbstractFloat}=SVector(cos(θ), sin(θ)) 
    #add and subtract to get new positions (they have the smallest radius)
    fp1::SVector{2, <: AbstractFloat}= @. pos1 + (unit_vector*model.min_radius)
    sp1::SVector{2, <: AbstractFloat}= @. pos1 - (unit_vector*model.min_radius)
    #mom is moved the fp1 and the daughter is added at sp1
    new_pos = fp1
    agent.r = model.min_radius
    agent.mass = model.min_mass
    agent.pos=Tuple(new_pos)
    # !!! IMPORTANT: Update positions in the CellListMap.PeriodicSystem
    @inbounds model.system.positions[agent.id] = SVector{2,AbstractFloat}(agent.pos)
    add_cell!(sp1,agent.kill_rate,agent.color,model)
end
function add_cell!(position::SVector{2, <: AbstractFloat},k_r::AbstractFloat,color::Symbol,model::ABM)
    id = last(model.avaliable_ids)
    pop!(model.avaliable_ids)
    #the agent is already in the system, we just need to change it now
    @inbounds agent=model[id]    
    agent.r=model.min_radius
    agent.mass=model.min_mass
    agent.kill_rate=k_r
    agent.IsPresent=true
    agent.color = color 
    @inbounds model.system.positions[id]=position  # bring that cell back into an active position real
    model.number_of_particles +=1
    agent.pos=Tuple(position)
    if agent.color==:blue
        model.blues+=1
    else
        model.reds+=1
    end
    return nothing
end
function wall_force(cur_pos::SVector{2, T}, agent::Particle, wall_constant::Number,sides::SVector{2,T})::SVector{2, T} where {T<: AbstractFloat}
    # Initialize force to zero
    force = SVector{2,T}(0.0, 0.0)
    # Check left wall
    overlap_left = agent.r - cur_pos[1]
    if overlap_left > 0
        force += SVector(wall_constant * overlap_left, 0.0)
    end
    # Check right wall
    overlap_right = cur_pos[1] + agent.r - sides[1]
    if overlap_right > 0
        force += SVector(-wall_constant * overlap_right, 0.0)
    end
    # Check bottom wall
    overlap_bottom = agent.r - cur_pos[2]
    if overlap_bottom > 0
        force += SVector(0.0, wall_constant * overlap_bottom)
    end
    # Check top wall
    overlap_top = cur_pos[2] + agent.r - sides[2]
    if overlap_top > 0
        force += SVector(0.0, -wall_constant * overlap_top)
    end
    return force
end
function simulate(model::ABM; nsteps::Integer=1,relaxation::Bool=false,early_stop::Bool=false,save_every::Integer=1)
    if relaxation
        model.relaxation=true
        step!(
            model, dummystep, model_step!, 100, false,
        ) 
        model.relaxation=false
    end
    if ! early_stop && (save_every==0)
    step!(
        model, dummystep, model_step!, nsteps, false,
    ) #the false makes it such that the model_step is taken first
    else
        x(agent) = agent.pos[1]
        y(agent) = agent.pos[2]
        r(agent) = agent.r
        IsPresent(agent) = agent.IsPresent
        color(agent) = agent.color           
        adata = [x, y, r, IsPresent, color]
        agent_data=init_agent_dataframe(model,adata)
        collect_agent_data!(agent_data,model,adata,0)
        for i in 1:nsteps
            step!(model,dummystep,model_step!,1,false)
            if i%save_every==0
                collect_agent_data!(agent_data,model,adata,i)
            end
            if early_stop && ((model.reds==model.carrying_capacity) || (model.blues==model.carrying_capacity))
                if ! (i%save_every==0)
                    collect_agent_data!(agent_data,model,adata,i)
                end
                break
            end
        end
        return agent_data 
    end
end
function CircleShape(h, k, r)
    θ = LinRange(0, 2π, 500)
    return h .+ r*sin.(θ), k .+ r*cos.(θ)
end
function animation_better(agent_data::DataFrame, sides::SVector{2,<: Number})
    steps_to_do = unique(agent_data.step)
    agent_gdf = groupby(agent_data, [:step])
    anim = @animate for step_idx in eachindex(steps_to_do)
        cur_df = agent_gdf[step_idx]
        #get rid of agents not IsPresent
        cur_df = filter(x -> x.IsPresent, cur_df)
        #get number of alive
        num = size(cur_df)[1]
        p = plot(xlim = (0.0, sides[1]), ylim = (0.0, sides[2]), grid = false, size = (350, 350), title = "Time = $(steps_to_do[step_idx]) Living cells = $num")
        for agent_idx in 1:size(cur_df)[1]
            x1, y1, r1, color = cur_df[agent_idx, [:x, :y, :r, :color]]
            plot!(p, CircleShape(x1, y1, r1), seriestype = [:shape], c = color, fillalpha = 0.2, legend = false)
        end
        # Turn off x and y tick markers and labels
    end
    return anim
end
function animation(agent_data::DataFrame,sides::SVector{2, <: Number}; coef::Integer=10)
    steps_to_do = unique(agent_data.step)
    agent_gdf = groupby(agent_data, [:step])
    anim = @animate for step_idx in eachindex(steps_to_do)
        cur_df = agent_gdf[step_idx]
        #get rid of agents not IsPresent
        cur_df = filter(x -> x.IsPresent, cur_df)
        #get number of alive
        num = size(cur_df)[1]
        # Plot cells 
        @df cur_df scatter(:x, :y, markersize=coef*:r, label=false,
                        color=:color, markerstrokewidth=0.1, xlim = (0.0, sides[1]), ylim = (0.0, sides[2]), grid = false, size = (350, 350), 
                        title = "Time = $(steps_to_do[step_idx]) Cells = $num",ticks=false)
    end
    return anim
end
function base_graphs(agent_data::DataFrame;file::String="")
    steps_to_do = unique(agent_data.step)
    agent_gdf = groupby(agent_data, [:step])
    data=zeros(length(steps_to_do))
    data2=zeros(length(steps_to_do))
    data3=zeros(length(steps_to_do))
    for step_idx in eachindex(steps_to_do)
        cur_df = agent_gdf[step_idx]
        #get rid of agents not IsPresent
        cur_df = filter(x -> x.IsPresent, cur_df)
        #get number of alive
        num_a = size(cur_df)[1]
        #get number of reds 
        num_r=size(filter(x -> x.color==:red, cur_df))[1]
        #get number of blues
        num_b = num_a - num_r
        data[step_idx]=num_a
        data2[step_idx]=num_r
        data3[step_idx]=num_b
    end
    if file != ""
        df=DataFrame(
            total=data,
            resistant=data2,
            sus=data3
        )
        CSV.write(file*".csv",df)
    end
    #p1=plot(data,label="Total",xlabel="Time Steps",ylabel="Population")
    p1=plot(data2,label="Faster Killing Strain",color=:red)
    p1=plot!(data3,label="Slower Killing Strain",color=:blue)
    plot(p1)
end
