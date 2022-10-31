import numpy as np
# physical/external base state of all entites
class EntityState(object): #实体状态，包括位置和速度
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance #agent的状态增加交流动作
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action交流行为
        self.c = None

# properties and state of physical world entity
# 是landmark，agent，border的父类
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with otherscollide
        self.collide = True
        # material density (affects mass)
        self.density = 25.0 
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None  # 表示加速度？ environment.py line 189调用
        # state 包括位置和速度
        self.state = EntityState()
        # mass 质量
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()
# 打卡点类
class Check(Entity):
    def __init__(self):
        super(Check, self).__init__()

# 边界类
class Border(Entity):
    def __init__(self):
        super(Border, self).__init__()
        self.pos = None

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals 可不可以交流
        self.silent = False
        # cannot observe the world 可不可以观察
        self.blind = False
        # physical motor noise amount 动作噪声
        self.u_noise = None
        # communication noise amount 交流噪声
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state   agent状态，AgentState()->EntityState()
        self.state = AgentState()
        # action
        self.action = Action() # agent.action包含Action类中的physical action ：agent.action.u 和communication a
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.index = 0
        self.landmarks = []
        self.borders = []
        # self.border = []  #原设置
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping 物理阻尼，对速度的一个影响
        self.damping = 0.5
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        self.buffer = []
    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.borders+self.check

    # return all agents controllable by external policies
    def border(self):
        return self.borders
    def landmark(self):
        return self.landmarks
    def check(self):
        return self.check
    def agent(self):
        return self.agents
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents: 
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)

        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                # noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                # p_force[i] = agent.action.u + noise # 将动作加上噪声
                p_force[i] = agent.action.u
                # p_force[i] = (agent.mass * agent.accel if agent.accel is not None else agent.mass) * agent.action.u + noise

        return p_force

    # 将agent的运动加在状态上，需要通过get_collision_force判断是不是碰撞
    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        # for a,entity_a in enumerate(self.entities):
        for a,entity_a in enumerate(self.agents): #
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        # for i,entity in enumerate(self.entities): # 如果是运动的实体则执行循环体，否则continue
        for i,entity in enumerate(self.agents): # 也就是只有是agent才执行循环体
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping) # 否则更新agent速度：速度×（1-阻尼）
            if (p_force[i] is not None):
                # entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                entity.state.p_vel += (p_force[i] / entity.mass - entity.state.p_vel * self.damping / entity.mass) * self.dt

            # No need to see
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k # numpy.logaddexp(x1, x2[, out])==log(exp(x1)+exp(x2))
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]