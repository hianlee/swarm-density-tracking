using Distributions, Distances, Parameters, LinearAlgebra, Random, Statistics
using DelimitedFiles, CSV, DataFrames, LightGraphs, SimpleWeightedGraphs
using PyCall

# Test settings
iter_max = 100000
# Agent Settings
w = 2
c = 2
max_speed = 0.1                     # Speed of fast agents
max_speed_fast = 0.26                # Speed of slow agents
n_particles = 50                    # Number of agents
n_neighbours = [15] #range(1, stop=40, step=1)  # Number of neighbours
width = exp10.(range(0.6, stop=2.65, length=125)) # Width of operating space
boundaries = width/2
memory = [20] #range(0, stop=50, step=2) # Memory length
n_fs = [0] #range(0, stop=40, step=2)   # Number of fast agents
sensing_radius = 1                  # Sensing range

# Behaviour Settings
rep_radius_max = 6 #12 # Maximum repulsion radius
rep_radius_min = 2  # Minimum repulsion radius
d_def = 6           # Exponential constant

# Target Settings
n_targets = 1
target_speed = 0.15  # range(0.1, stop=0.26, step=0.02)
detection_radius = 1
turn_limit = 180
movement_policy = "ne" # [ne, e, mix] (ne - non-evasive, e - pure evasive, mix - evasive then attempts to outrun when cornered)
encounter_limit = 10
jump_timer = 30
rep_radius = 1


@with_kw mutable struct Particle
    # User defined parameters
    name::Int
    k::Int
    max_speed::Float64
    boundary::Float64
    memory::Int
    sensing_range::Float64
    w::Float64
    c::Float64
    rep_radius_max::Float64
    rep_radius_min::Float64
    rep_radius = rep_radius_min
    d::Float64

    # State parameters
    position::Vector = [rand(Uniform(-boundary, boundary)),
                        rand(Uniform(-boundary, boundary))]
    velocity::Vector = [0., 0.]
    waypoint = [rand(Uniform(-boundary, boundary)),
                rand(Uniform(-boundary, boundary))]

    timer = 0
    mode = "explore"
    density = 0
    neighbours = []

    # Own target information
    target_pos = []
    timestamp = 0

    # Target information from neighbours
    n_target_pos = []
    n_timestamp = 0

    # Used information
    attraction_point = []
    used_timestamp = 0

    # Movement Velocities
    attract_vel = [0., 0.]
    repel_vel = [0., 0.]
end

# Agent functions
function update_target_info!(self::Particle, target_list::Array)
    for target in target_list
        # If directly detecting target
        if norm(self.position - target.position) <= self.sensing_range
            self.mode = "track"
            self.attraction_point = self.target_pos = target.position
            self.used_timestamp = self.timestamp = self.timer
            break
        # If own memory time out, clear memory
        elseif self.target_pos != [] && self.timer > (self.timestamp +
            self.memory)
            self.mode = "explore"
            self.target_pos = []
            self.timestamp = 0
        end
    end
end

function get_neighbour_info(self::Particle, agent_list::Array)
    stored_neighbours = 0
    n_target_pos = []
    n_times = zeros(self.k)
    n_distances = zeros(self.k)
    n_positions = []
    modes = []
    density_distances = []
    n_names = []
    n = 6   # Number of nearest neighbours to consider when calculating local density

    for agent in agent_list
        if agent.name == self.name
            continue
        else
            distance = norm(self.position - agent.position)

            # This block to gather distances between n neighbours to calculate local density
            if length(density_distances) < n
                push!(density_distances, distance)
            elseif distance < maximum(density_distances)
                density_distances[argmax(density_distances)] = distance
            end

            # This block to gather communications neighbours
            if stored_neighbours < self.k
                stored_neighbours += 1
                push!(n_positions, agent.position)
                push!(modes, agent.mode)
                n_distances[stored_neighbours] = distance
                push!(n_target_pos, agent.target_pos)
                n_times[stored_neighbours] = agent.timestamp
                push!(n_names, agent.name)
            else
                if distance < maximum(n_distances)
                    index = argmax(n_distances)
                    n_distances[index] = distance
                    n_positions[index] = agent.position
                    modes[index] = agent.mode
                    n_target_pos[index] = agent.target_pos
                    n_times[index] = agent.timestamp
                    n_names[index] = agent.name
                end
            end
        end
    end

    self.neighbours = n_names
    avg_density_distance = mean(density_distances)
    self.density = (n + 1)/(pi * avg_density_distance ^ 2)

    return n_target_pos, n_times, n_positions
end

function set_target!(self::Particle, n_target_pos::Array, n_times::Array)
    most_recent_ts = maximum(n_times)

    # Neighbour's information supersedes own information
    if most_recent_ts > self.timestamp
        # Delete own information
        self.target_pos = []
        self.timestamp = 0

        # Save neighbour information
        index = argmax(n_times)
        self.n_target_pos = self.attraction_point = n_target_pos[index]
        self.n_timestamp = self.used_timestamp = n_times[index]
        self.mode = "track"
    # If neighbour's information is old, use own information
    elseif self.target_pos != []
        self.mode = "track"
        self.attraction_point = self.target_pos
        self.used_timestamp = self.timestamp
        self.n_target_pos = []
        self.n_timestamp = 0
    else
        self.mode = "explore"
        self.attraction_point = []
        self.used_timestamp = 0
        self.n_target_pos = []
        self.n_timestamp = 0
    end
end

function integrate_neighbour_info!(self::Particle, agent_list::Array)
    n_target_pos, n_times, n_positions = get_neighbour_info(self, agent_list)
    set_target!(self, n_target_pos, n_times)
    return n_positions
end

function clear_memory!(self::Particle)
    if self.target_pos != [] && self.timer > (self.timestamp + self.memory)
        self.target_pos = []
        self.timestamp = 0
    end
    if self.n_target_pos != [] && self.timer > (self.n_timestamp + self.memory)
        self.n_target_pos = []
        self.n_timestamp = 0
    end
    if self.attraction_point != [] && self.timer > (self.used_timestamp + self.memory)
        self.mode = "explore"
        self.timestamp = self.n_timestamp = self.used_timestamp = 0
        self.attraction_point = []
    end
end

function set_attraction_velocity!(self::Particle)
    if self.attraction_point != []
        self.attract_vel = (self.w * self.velocity) + (self.c * rand(1)) .*
                            (self.attraction_point - self.position)
    else
        self.attract_vel = self.w .* self.velocity
    end
end

function set_repulsion_velocity!(self::Particle, n_positions::Array)

    self.repel_vel = [0., 0.]

    # Constant Repulsion Block
    # self.rep_radius = self.rep_radius_min

    # Adaptive Repulsion Block
    if self.mode == "explore" && self.rep_radius < self.rep_radius_max
        self.rep_radius += 0.1
    elseif self.mode == "track" && self.rep_radius > self.rep_radius_min
        self.rep_radius -= 0.75
    end

    s = pi * self.rep_radius ^ 2
    alpha_r = sqrt(s / self.k)

    for position in n_positions
        vector = self.position - position
        distance = norm(vector)
        self.repel_vel += ((alpha_r / distance) ^ self.d) * (vector / distance)
    end
end

function move_agent!(self::Particle)
    # Speed limiter
    self.velocity = self.attract_vel + self.repel_vel
    speed = norm(self.velocity)
    if speed > self.max_speed
        self.velocity = (self.max_speed / speed) * self.velocity
    end

    # Boundary conditions
    waypoint = self.position + self.velocity
    if abs(waypoint[1]) > self.boundary
        if waypoint[1] > self.boundary
            waypoint[1] = self.boundary
        else
            waypoint[1] = -self.boundary
        end
    end

    if abs(waypoint[2]) > self.boundary
        if waypoint[2] > self.boundary
            waypoint[2] = self.boundary
        else
            waypoint[2] = -self.boundary
        end
    end

    self.position = waypoint

end


@with_kw mutable struct Target
    # User defined settings
    name::String
    sensing_range::Float64
    max_speed::Float64
    boundary::Float64

    # Evasion settings
    d::Float64
    rep_radius::Float64
    policy::String

    # Movement Setting
    encounter::Int = 0
    encounter_limit::Int
    jump_timer::Int
    turn_limit::Float64
    timer = 0
    tracked = false

    # State Parameters
    position::Vector = [rand(Uniform(-boundary, boundary)),
                        rand(Uniform(-boundary, boundary))]
    heading::Float64 = rand(Uniform(-180, 180))
    velocity::Vector = [cosd(heading) * max_speed, sind(heading) * max_speed]
    waypoint = [rand(Uniform(-boundary, boundary)),
                rand(Uniform(-boundary, boundary))]
    evade = false
end

# Target Functions
function find_agents(self::Target, agent_list::Array)
    agent_pos = []
    in_range = 0
    for agent in agent_list
        if norm(self.position - agent.position) <= self.sensing_range
            in_range += 1
            push!(agent_pos, agent.position)
        end
    end
    if in_range > 0
        self.tracked = true
    else
        self.tracked = false
    end

    return agent_pos, in_range
end

function evasive_velocity!(self::Target, agent_pos::Array, in_range::Int)
    self.velocity = [0., 0.]
    if in_range > 0
        self.encounter += 1

        s = pi * self.rep_radius ^ 2
        alpha_r = sqrt(s / in_range)
        for position in agent_pos
            vector = self.position - position
            distance = norm(vector)
            self.velocity += ((alpha_r / distance) ^ self.d) *
                                (vector / distance)
        end

        self.heading = atand(self.velocity[2], self.velocity[1])
        speed = norm(self.velocity)

        if speed > self.max_speed
            self.velocity = (self.max_speed / speed) * self.velocity
        end

    else
        self.velocity = [self.max_speed * cosd(self.heading),
                        self.max_speed * sind(self.heading)]

        if self.encounter > 0
            self.encounter -= 1
        end
    end
end

function non_evasive_velocity!(self::Target)
    vec_to_wp = self.waypoint - self.position
    if norm(vec_to_wp) <= (1.5) || self.timer >= 200
        self.waypoint = [rand(Uniform(-self.boundary, self.boundary)),
                            rand(Uniform(-self.boundary, self.boundary))]
        self.timer = 0
    else
        self.timer += 1
    end

    req_heading = atand(vec_to_wp[2], vec_to_wp[1])
    heading_delta = req_heading - self.heading

    if heading_delta > 180
        heading_delta -= 360
    elseif heading_delta < -180
        heading_delta += 360
    end

    if abs(heading_delta) > self.turn_limit
        if heading_delta > 0
            self.heading += self.turn_limit
        else
            self.heading -= self.turn_limit
        end
    else
        self.heading = req_heading
    end

    self.velocity = [self.max_speed * cosd(self.heading),
                        self.max_speed * sind(self.heading)]
end

function move_target!(self::Target, agent_list::Array)
    agent_pos, in_range = find_agents(self, agent_list)
    if self.policy == "ne"
    non_evasive_velocity!(self)
    elseif self.policy == "e"
        evasive_velocity!(self, agent_pos, in_range)
    elseif self.policy == "mix" && self.encounter < self.encounter_limit
        evasive_velocity!(self, agent_pos, in_range)
    elseif self.policy == "mix" &&
            (self.encounter >= self.encounter_limit || self.evade == false)
        non_evasive_velocity!(self)
        self.jump_timer -= 1
        if self.jump_timer != 0
            self.evade = false
        else
            self.encounter = 0
            self.evade = true
            self.jump_timer = 30
        end
    end

    self.position += self.velocity

    # Reflect off boundaries
    if abs(self.position[1]) > self.boundary
        if self.position[1] > self.boundary  # Right boundary
            overshoot = self.position[1] - self.boundary
            self.position[1] -= overshoot
            if self.heading > 0
                self.heading = 180 - self.heading
            else
                self.heading = -180 - self.heading
            end

        elseif self.position[1] < -self.boundary  # Left boundary
            overshoot = self.position[1] + self.boundary
            self.position[1] -= overshoot
            if self.heading > 0
                self.heading = 180 - self.heading
            else
                self.heading = -180 - self.heading
            end
        end
    end

    if abs(self.position[2]) > self.boundary
        if self.position[2] > self.boundary  # Top boundary
            overshoot = self.position[2] - self.boundary
            self.position[2] -= overshoot
        elseif self.position[2] < -self.boundary  # Bottom boundary
            overshoot = self.position[2] + self.boundary
            self.position[2] -= overshoot
        end
        self.heading = -self.heading
    end
end

all_score = []
all_proportions = []
all_loc_density = []
all_clustering = []

for boundary in boundaries   # for different environment sizes
    global iter_max, c, max_speed, n_particles, memory, n_fs
    sensing_radius, repulsion, rep_radius_max, rep_radius_min, d_def, n_targets,
    target_speed, detection_radius, turn_limit, movement_policy, encounter_limit,
    jump_timer, rep_radius, n_neighbours, w

    k = n_neighbours[1]
    nf = n_fs[1]
    time_limit = memory[1]
    println("Start k", k, " Memory ", time_limit, " Fast Agents ", nf)
    println("Width: ", boundary*2)
    Random.seed!(5512)

    score = 0

    pos_history::Vector{Array{Array{String,1},1}} = []
    vel_history::Vector{Array{Array{Float64,1},1}} = []
    target_history::Vector{Array{Array{Float64,1},1}} = []
    exploit_proportion = []
    average_loc_density = []
    clustering_current = []

    agent_list = []
    target_list = []

    for i in range(1, stop=n_particles-nf)
        name = i
        agent = Particle(name=name, k=k, max_speed=max_speed, boundary=boundary,
                        memory=time_limit, sensing_range=sensing_radius, w=w,
                        c=c, rep_radius_max=rep_radius_max,
                        rep_radius_min=rep_radius_min, d=d_def)
        agent.boundary -= sensing_radius/2
        push!(agent_list, agent)
    end

    ## Enable this block for heterogeneous swarm
    ## !!Warning: Even if nf == 0, this block will spawn a fast agent if uncommented!!
    ## !!Do not enable this block if you want a homogeneous swarm!!
    # for i in range(n_particles-nf, stop=n_particles)
    #     name = i
    #     agent = Particle(name=name, k=k, max_speed=max_speed_fast, boundary=boundary,
    #                     memory=time_limit, sensing_range=sensing_radius, w=w,
    #                     c=c, rep_radius_max=rep_radius_max,
    #                     rep_radius_min=rep_radius_min, d=d_def)
    #     agent.boundary -= sensing_radius/2
    #     push!(agent_list, agent)
    # end

    for i in range(1, stop=n_targets)
        name = "Target " * string(i)
        target = Target(name=name, sensing_range=detection_radius,
                        max_speed=target_speed, boundary=boundary, d=d_def,
                        rep_radius=rep_radius, policy=movement_policy,
                        encounter_limit=encounter_limit, jump_timer=jump_timer,
                        turn_limit=turn_limit)

        if target.policy == "mix" || target.policy == "e"
            target.evade = true
        else
            target.evade = false
        end
        push!(target_list, target)
    end

    iteration = 0

    while iteration < iter_max

        write_pos_current = []
        pos_current = []
        vel_current = []
        target_current = []
        iter_exploit_proportion = 0
        loc_density = 0

        for target in target_list
            move_target!(target, agent_list)
            push!(target_current, target.position)
            if target.tracked
                score += 1
            end
        end
        push!(target_history, target_current)

        for agent in agent_list
            # This saves tt, sf, and ss onto the position file so that the video processing file knows if an agent is tracking or not
            # Also differentiates between frast agents and slow agents
            if agent.mode == "track"
                save_pos = ["tt" * string(agent.position[1]), string(agent.position[2])]
                iter_exploit_proportion += 1
            elseif agent.max_speed == max_speed_fast
                save_pos = ["sf" * string(agent.position[1]), string(agent.position[2])]
            else
                save_pos = ["ss" * string(agent.position[1]), string(agent.position[2])]
            end
            update_target_info!(agent, target_list)
            clear_memory!(agent)
            push!(pos_current, [agent.position])
            push!(write_pos_current, save_pos)
            push!(vel_current, agent.velocity)
        end
        push!(pos_history, write_pos_current)
        push!(vel_history, vel_current)

        push!(exploit_proportion, iter_exploit_proportion/n_particles)

        for agent in agent_list
            n_positions = integrate_neighbour_info!(agent, agent_list)

            set_attraction_velocity!(agent)
            set_repulsion_velocity!(agent, n_positions)

            loc_density += agent.density

        end

        push!(average_loc_density, loc_density/n_particles)

        for agent in agent_list
            move_agent!(agent)
            agent.timer += 1
        end

        iteration += 1
        # println(iteration)
    end

    score /= (iter_max * n_targets)
    println("k = ", k, " fast agents: ", nf, " score: ", score)
    # println("Iter Exploit Proportion: ", mean(exploit_proportion))
    push!(all_score, score)
    push!(all_proportions, mean(exploit_proportion))
    push!(all_loc_density, mean(average_loc_density))

    file_name1 = "julia_test_target.csv"
    file_name2 = "julia_test_agents.csv"
    file_name3 = "julia_test_vels.csv"
    CSV.write(file_name1, DataFrame(target_history), writeheader=false)
    CSV.write(file_name2, DataFrame(pos_history), writeheader=false)
    CSV.write(file_name3, DataFrame(vel_history), writeheader=false)
end

score_file = "non_evasive_20_mem_score.csv"
proportion_file = "non_evasive_20_mem_engagement.csv"
density_file = "50agents_ne_k15_v15_n3_density.csv"
network_file = "50agents_ne_k15_v15_n3_network.csv"

# CSV.write(score_file, DataFrame([all_score]), writeheader=false)
# CSV.write(proportion_file, DataFrame([all_proportions]), writeheader=false)
# CSV.write(density_file, DataFrame([all_loc_density]), writeheader=false)
# CSV.write(network_file, DataFrame([all_clustering]), writeheader=false)


println("Done")
