"""
Author: Isaac Sheidlower, AABL Lab, Isaac.Sheidlower@tufts.edu
Github: https://github.com/IsaacSheidlower
"""

import os, glob
import functools
from typing import MutableMapping
import cv2
import csv

from flask import Flask, request, render_template, Response, redirect, session,send_from_directory, redirect
from flask_wtf import FlaskForm
from wtforms import SubmitField

import numpy as np
import gym
from gym import wrappers
import torch

import brax 
from IPython.display import HTML, Image 
from brax import envs
from brax import jumpy as jp
from brax.envs import to_torch
from brax.io import html
from brax.io import image
import jax
from jax import numpy as jnp

import time, copy

import logging

from PIL import Image

from PS_utils import sample_normal
from sac_torch import Agent
from dqn_agent import Agent as TamerAgent

import gym, random, pickle, os.path, math, glob
import stopwatch as SW

from uid.uuid_manager import uuid_manager
uuid_manager = uuid_manager()

from functools import wraps

import multiprocessing

from sys import platform
if platform == "linux" or platform == "linux2":
    os.environ["SDL_VIDEODRIVER"] = "dummy" # For headless linux servers

# os.environ['DISPLAY'] = ":0"
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(DIR_PATH, 'templates/')

#from pyvirtualdisplay import Display

#virtual_display = Display(visible=0, size=(1400, 900))
#virtual_display.start()


global finished 
finished = False

app = Flask(__name__, template_folder=TEMPLATE_PATH, static_folder="static")
app.secret_key = 'test key'


# logging.basicConfig(filename='record.log', level=logging.DEBUG, format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

@app.route("/")
def index():
    # First do GDPR and Age checks
    return render_template("gdpr_age.html")


@app.route("/qualified", methods=["POST"])
def qualified():
    #active = np.genfromtxt("session_is_active.csv", delimiter=',')
    #if active == 1.0:
    #    return render_template("server_is_busy.html")
    qual = request.form.get('age18', False) and request.form.get('gdpr', False)
    if (qual):
        # generate a unique uuid to offer to users
        #uuid = uuid_manager.get_uuid()
        app.logger.debug('qualified successfully', qual)
        uuid = str(np.random.randint(low=10000, high=100000000))
        return redirect("/app/{}".format(uuid), code=302)
    else:
        return render_template("thanks.html")

@app.route("/consent", methods=["POST"])
def consent():
    uuid = session['uuid']
    user_info = {}
    user_info['agreed'] = request.form.get('agreed', False)
    session[uuid] = user_info
    app.logger.debug('provided consent', user_info['agreed'])
    #print(session['uuid'])
    return render_template("info_lunar_toggle.html")

@app.route("/info", methods=["POST"])
def info():
    return redirect("/app/{}".format(session['uuid']), code=302)
    
@app.route("/", methods=['GET', 'POST'])
@app.route("/app/<uuid>", methods=['GET', 'POST'])
def index_uuid(uuid):
    if uuid in session:
        if session[uuid].get('agreed'):
            
            if np.random.random()<(1/3):
                feed_filename = f"tamer_feedback_data/participant_feed_{uuid}.csv"
                with open(feed_filename, "x", newline='') as csvfile:
                    pass
                    #print("Done")
                tamer_learning_loop(uuid)
                return render_template("tamer_simulation.html", render=f"render_{uuid}", uuid=uuid)
            else:
                feed_filename = f"feedback_data/participant_feed_{uuid}.csv"
                with open(feed_filename, "x", newline='') as csvfile:
                    pass
                    #print("Done")
                cair_learning_loop(uuid)
                return render_template("simulation.html", render=f"render_{uuid}", uuid=uuid)
            
        else:
            return render_template("excluded_participant.html")
    session['uuid'] = uuid
    return render_template("consent.html", missing_data = uuid in session)


@app.route('/send_good_feedback/<uuid>')
def send_good_feedback(uuid):
    #filename = "participant_data.csv"
    with open(f"feedback_data/participant_feed_{uuid}.csv", 'a', newline='') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(["1", time.time()]) 
        csvfile.close()
    return "nothing"

@app.route('/send_bad_feedback/<uuid>')
def send_bad_feedback(uuid):
    #filename = "participant_data.csv"
    with open(f"feedback_data/participant_feed_{uuid}.csv", 'a', newline='') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(["-1", time.time()]) 
        csvfile.close()
    return "nothing"

@app.route('/send_no_feedback/<uuid>')
def send_no_feedback(uuid):
    #filename = "participant_data.csv"
    with open(f"feedback_data/participant_feed_{uuid}.csv", 'a', newline='') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(["0", time.time()]) 
        csvfile.close()
    return "nothing"

@app.route('/tamer_send_good_feedback/<uuid>')
def tamer_send_good_feedback(uuid):
    #filename = "participant_data.csv"
    with open(f"tamer_feedback_data/participant_feed_{uuid}.csv", 'a', newline='') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(["1", time.time()]) 
        csvfile.close()
    return "nothing"

@app.route('/tamer_send_bad_feedback/<uuid>')
def tamer_send_bad_feedback(uuid):
    #filename = "participant_data.csv"
    with open(f"tamer_feedback_data/participant_feed_{uuid}.csv", 'a', newline='') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(["-1", time.time()]) 
        csvfile.close()
    return "nothing"

@app.route('/tamer_send_no_feedback/<uuid>')
def tamer_send_no_feedback(uuid):
    #filename = "participant_data.csv"
    with open(f"tamer_feedback_data/participant_feed_{uuid}.csv", 'a', newline='') as csvfile:  
        csvwriter = csv.writer(csvfile)  
        csvwriter.writerow(["0", time.time()]) 
        csvfile.close()
    return "nothing"

def frame_gen(env_func, *args, **kwargs):
    get_frame = env_func(*args, **kwargs)
    while finished == False:
        frame = next(get_frame, None)
        if frame is None:
            break
        _, frame = cv2.imencode('.png', frame)

        frame = frame.tobytes()
        yield (b'--frame\r\n' + b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

def render_browser(env_func):
        def wrapper(*args, **kwargs):
            @wraps(wrapper)
            @app.route(f"/render_feed/{session['uuid']}", endpoint=f"render_{session['uuid']}")
            def render_feed():
                return Response(frame_gen(env_func, *args, **kwargs), mimetype='multipart/x-mixed-replace; boundary=frame')
        return wrapper


@render_browser
def cair_learning_loop(uuid, end=False):

    if end:
        img = Image.open("thankYou2.jpg")
        arr = np.array(img)
        yield arr
        return True
    class ActionQueue:
        def __init__(self, size=5):
            self.size = size
            self.queue = []

        def enqeue(self, action_memory):
            if len(self.queue) > self.size:
                self.queue.pop()
                self.queue.insert(0, action_memory)
            else:
                self.queue.insert(0, action_memory)

        def push_to_buffer_and_learn(self, agent, actor, tf, feedback_value, interval_min=0, interval_max=.8):
            if len(self.queue) == 0: return None
            else:
                avg_loss = []
                i = 0
                for action_memory in self.queue: 
                    b = tf - action_memory[2]
                    a = tf - action_memory[3]
                    #print(b, a)
                    # feedback must occur within 0.2-4 seconds after feedback to have non-zero importance weight
                    #print(a, b)
                    pushed = (b >= interval_min and a <= interval_max)
                    #print(tf)
                    # push: state, action, ts, te, tf, feedback
                    # [observation, action, ts, te, tf, feedback_value, mu, sigma, old_observation, done]
                    if pushed:
                        #print("AAAAAA")
                        agent.remember(action_memory[8], action_memory[1], feedback_value, action_memory[0], action_memory[9])
                    i += 1
                return np.mean(np.array(avg_loss))
            
        def clear(self):
            self.queue = []

    #env = gym.make('BipedalWalker-v3')
    
    #env = gym.make('LunarLanderContinuous-v2')
    #env.viewer = None

    environment = "inverted_pendulum"  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'reacher', 'walker2d', 'fetch', 'grasp', 'ur5e']
    env = envs.create(env_name=environment)
    state = env.reset(rng=jp.random_prngkey(seed=0))
    #state = jax.jit(env.step)(state, jnp.ones((env.action_size,)))
    #env = wrappers.Monitor(env, "media", video_callable=False, force=True)
    
    # entry_point = functools.partial(envs.create_gym_env, env_name='reacher')
    # if 'brax-reacher-v0' not in gym.envs.registry.env_specs:
    #     gym.register('brax-reacher-v0', entry_point=entry_point)

    # env = gym.make("brax-reacher-v0", batch_size=1)

    # env = to_torch.JaxToTorchWrapper(env, device='cpu')


    rew_filename = f"reward_data/participant_{uuid}.csv"
    with open(rew_filename, "x", newline='') as csvfile:
        pass
        #print("Done")
        
    state_filename = f"state_data/participant_state_{uuid}.csv"
    with open(state_filename, "x", newline='') as csvfile:
        pass
        #print("Done")

    # Note: these credit assignment intervals impact how the agent behaves a lot.
    # Because of this sensitivity the model is overall very sensitive.
    # There is a frame time delay of .1 so teaching is not boring. Could make the 
    # the agent much better if played at around 10 frames per second (not sure of  current fps)
    interval_min = 0.1
    interval_max = 1.0

    episodes=500
    USE_CUDA = torch.cuda.is_available()
    learning_rate = .001
    replay_buffer_size = 100000
    learn_buffer_interval = 200  # interval to learn from replay memory
    batch_size = 200
    print_interval = 1000
    log_interval = 1000
    learning_start = 100
    #win_reward = 21     # Pong-v4
    win_break = True
    queue_size = 1000

    #print(env.observation_size)

    agent = Agent(alpha=.001, beta=.001, max_size=100000, layer1_size=64, layer2_size=64, input_dims=(5,), env=env,
                n_actions=env.action_size, reward_scale=10)

    actor = Agent(alpha=.001, beta=.001, layer1_size=64, layer2_size=64, input_dims=(5,), env=env,
                n_actions=env.action_size, reward_scale=2, auto_entropy=False)
    
    np_state = np.array(state.obs)
    observation = [np_state[0], np_state[1], np_state[5], np_state[6], np_state[9]]

    episode_rewards = []
    all_rewards = []
    sum_rewards=[]
    losses = []
    episode_num = 0
    is_win = False

    stopwatch = SW.Stopwatch()

    cnt=0
    start_f=0
    end_f=0

    action_queue = ActionQueue(queue_size)
    
    kappa_possibilities = [0.8, 0.9]
    kappa = kappa_possibilities[np.random.randint(low=0, high=2)]
    #print(kappa, "KAPPA")

    rewards = []
    e = 0.1
    render = True
    
    timeout = time.time() + 60*15  # length of interaction

    true_done = False
    tf = 0
    while time.time() < timeout and true_done == False:
        #print(uuid)
        start_f=end_f
        stopwatch.restart()
        loss = 0

        #state = env.reset(rng=jp.random_prngkey(seed=0))
        
        action = jp.zeros((env.action_size,))
        #state = jax.jit(env.step)(state, action)
        #state = env.step(state, action)
        state = jax.jit(env.step)(state, action)
        #yield image.render_array(env.sys, state.qp, width=500, height=500)
        np_state = np.array(state.obs)
        observation = [np_state[0], np_state[1], np_state[5], np_state[6], np_state[9]]
        reward = state.reward
        done = state.done
        yield image.render_array(env.sys, state.qp, width=500, height=500, ssaa=1)

        #yield scene
        #time.sleep(3)
        # observation = env.reset()
        # observation = observation.numpy()[0]
        # print(observation)
        ep_rewards = 0
        feedback_value = 0
        #tf_old = 0
        time.sleep(1.0)
        while(True):
            #print(feedback_value)
            stopwatch.start()
            #scene = env.render(mode='rgb_array')
            #yield scene
            #action = jp.ones((env.action_size,)) * jp.sin(end_f * jp.pi / 15)
            #state = jax.jit(env.step)(state, action)
            #state = env.step(state, action)
            #state = jax.jit(env.step)(state, action)
            #yield image.render_array(env.sys, state.qp, width=400, height=400)

            end_f+=1
            ts = stopwatch.duration
            ts = time.time()
            action, dist, mu, sigma = sample_normal(agent, actor, observation, with_noise=False, max_action=1, kappa=kappa)
            #action, dist, mu, sigma = agent.sample_action(observation)
            #print(action)
            old_observation = observation

            #observation, reward, done, _ = env.step(torch.tensor(np.array(action[None,:])))
            #observation = observation.numpy()[0]
            #state = env.step(state, action)
            state = jax.jit(env.step)(state, action)
            
            scene_img =  image.render_array(env.sys, state.qp, width=500, height=500, ssaa=1)
            yield scene_img
            np_state = np.array(state.obs)
            observation = [np_state[0], np_state[1], np_state[5], np_state[6], np_state[9]]
            reward = state.reward
            done = state.done
            #print("AAAA")
            
            actor.remember(old_observation, action, reward, observation, done)
            episode_rewards.append(reward)
            ep_rewards += reward
            te = time.time()
            #feedback_value = 0 # uncomment for sparse feedback
            time.sleep(e) # Delay to make the game seeable
            feedback = ""
            all_feedback = np.genfromtxt(f"feedback_data/participant_feed_{uuid}.csv", delimiter=',')

            if len(all_feedback) > 0:
                if len(all_feedback) == 2 and np.size(all_feedback) == 2:
                    latest_feedback = all_feedback
                else:
                    latest_feedback = all_feedback[len(all_feedback)-1] 
                    feedback = latest_feedback[0]
                    tf_old = copy.copy(tf) 
                    tf = latest_feedback[1]
                    if tf != tf_old:
                        if feedback == 1:
                            feedback_value = 1
                            loss = action_queue.push_to_buffer_and_learn(agent, actor, tf, feedback_value, interval_min=interval_min, interval_max=interval_max)
                            with open(state_filename, 'a', newline='') as csvfile:  
                                csvwriter = csv.writer(csvfile)  
                                csvwriter.writerow([observation, tf]) 
                                csvfile.close()    
                        elif feedback == -1:
                            feedback_value = -1
                            loss = action_queue.push_to_buffer_and_learn(agent, actor, tf, feedback_value, interval_min=interval_min, interval_max=interval_max)
                            with open(state_filename, 'a', newline='') as csvfile:  
                                csvwriter = csv.writer(csvfile)  
                                csvwriter.writerow([observation, tf]) 
                                csvfile.close()    
                        #else: 
                            # DOUBLE CHECK
                            #feedback_value = 0 
                            #loss = action_queue.push_to_buffer_and_learn(agent, actor, tf, feedback_value, interval_min=interval_min, interval_max=interval_max)
            #print(tf, ts, tf-ts)
            action_queue.enqeue([observation, action, ts, te, tf, feedback_value, mu, sigma, old_observation, done])

            if feedback_value != 0:
                #tf = stopwatch.duration
                # [observation, action, ts, te, tf, feedback_value, mu, sigma, old_observation, done]
                loss = action_queue.push_to_buffer_and_learn(agent, actor, tf, feedback_value, interval_min=interval_min, interval_max=interval_max)
                #agent.remember(old_observation, action, feedback_value, observation, done)
                #print(loss, "feedback loss")

            actor.learn()
            agent.learn()

            #yield scene_img
            
            if stopwatch.duration > 60:
                done = True

            if done:
                with open(rew_filename, 'a', newline='') as csvfile:  
                    csvwriter = csv.writer(csvfile)  
                    csvwriter.writerow([episode_num, ep_rewards, end_f, time.time(), kappa]) 
                    csvfile.close()                                                         
                #print(ep_rewards)
                #print("Episode:", str(episode_num))
                rewards.append(ep_rewards)
                ep_rewards = 0
                episode_rewards = []
                episode_num += 1
                feedback_value = 0
                action_queue.clear()
                #yield image.render_array(env.sys, state.qp, width=500, height=500, ssaa=1)

                #if len(rewards) > 0:
                    #if rewards[len(rewards)-1] >= 400:
                        #true_done = True
                break

            if time.time() > timeout and done:
                true_done = True
                break         
            
    finished = True
    #env.close()
    del env, agent, actor, action_queue
    img = cv2.imread("cair_outro_slide_image.jpg")

    cv2.putText(img,str(uuid), (270, 550), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)
    
    arr = np.array(img)
    yield arr
    return True

@render_browser
def tamer_learning_loop(uuid, end=False):

    if end:
        img = Image.open("thankYou2.jpg")
        arr = np.array(img)
        yield arr
        return True

    class ActionQueue:
        def __init__(self, size=5):
            self.size = size
            self.queue = []

        def enqeue(self, action_memory):
            if len(self.queue) > self.size:
                self.queue.pop()
                self.queue.insert(0, action_memory)
            else:
                self.queue.insert(0, action_memory)

        def push_to_buffer(self, agent, tf, feedback_value, interval_min=0, interval_max=.8):
            if len(self.queue) == 0: return None
            else:
                avg_loss = []
                i = 0
                for action_memory in self.queue: 
                    b = tf - action_memory[2]
                    a = tf - action_memory[3]
                    #print(b, a)
                    # feedback must occur within 0.2-4 seconds after feedback to have non-zero importance weight
                    #print(a, b)
                    pushed = (b >= interval_min and a <= interval_max)
                    #print(tf)
                    # push: state, action, ts, te, tf, feedback
                    # [observation, action, ts, te, tf, feedback_value, old_observation, done]
                    if pushed:
                        #print("AAAAA")
                        agent.remember(action_memory[6], action_memory[1], feedback_value, action_memory[0], action_memory[7])
                    i += 1
                return np.mean(np.array(avg_loss))
            
        def clear(self):
            self.queue = []

    #env = gym.make('BipedalWalker-v3')
    
    #env = gym.make('LunarLanderContinuous-v2')
    #env.viewer = None

    environment = "inverted_pendulum"  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'reacher', 'walker2d', 'fetch', 'grasp', 'ur5e']
    env = envs.create(env_name=environment)
    state = env.reset(rng=jp.random_prngkey(seed=0))
    #state = jax.jit(env.step)(state, jnp.ones((env.action_size,)))
    #env = wrappers.Monitor(env, "media", video_callable=False, force=True)
    
    # entry_point = functools.partial(envs.create_gym_env, env_name='reacher')
    # if 'brax-reacher-v0' not in gym.envs.registry.env_specs:
    #     gym.register('brax-reacher-v0', entry_point=entry_point)

    # env = gym.make("brax-reacher-v0", batch_size=1)

    # env = to_torch.JaxToTorchWrapper(env, device='cpu')


    rew_filename = f"tamer_reward_data/participant_{uuid}.csv"
    with open(rew_filename, "x", newline='') as csvfile:
        pass
        #print("Done")
    state_filename = f"tamer_state_data/participant_state_{uuid}.csv"
    with open(state_filename, "x", newline='') as csvfile:
        pass
    # Note: these credit assignment intervals impact how the agent behaves a lot.
    # Because of this sensitivity the model is overall very sensitive.
    # There is a frame time delay of .1 so teaching is not boring. Could make the 
    # the agent much better if played at around 10 frames per second (not sure of  current fps)
    interval_min = 0.1
    interval_max = 1.0

    queue_size = 3

    #print(env.observation_size)

    agent = TamerAgent(state_size=5, action_size=2, seed=0)

    np_state = np.array(state.obs)
    observation = np.array([np_state[0], np_state[1], np_state[5], np_state[6], np_state[9]])


    episode_rewards = []

    episode_num = 0

    stopwatch = SW.Stopwatch()

    end_f=0

    action_queue = ActionQueue(queue_size)


    rewards = []
    e = 0.1
    render = True
    
    timeout = time.time() + 60*15  # length of interaction

    true_done = False
    tf = 0

    eps = .3
    min_eps = .1
    eps_decay = .999

    while time.time() < timeout and true_done == False:
        #print(uuid)
        start_f=end_f
        stopwatch.restart()
        loss = 0

        #state = env.reset(rng=jp.random_prngkey(seed=0))
        
        action = jp.zeros((env.action_size,))
        #state = jax.jit(env.step)(state, action)
        #state = env.step(state, action)
        state = jax.jit(env.step)(state, action)
        #yield image.render_array(env.sys, state.qp, width=500, height=500)
        np_state = np.array(state.obs)
        observation = np.array([np_state[0], np_state[1], np_state[5], np_state[6], np_state[9]])
        reward = state.reward
        done = state.done
        yield image.render_array(env.sys, state.qp, width=500, height=500, ssaa=1)


        ep_rewards = 0
        feedback_value = 0
        #tf_old = 0
        time.sleep(1)
        while(True):
            #print(feedback_value)
            stopwatch.start()


            end_f+=1
            ts = stopwatch.duration
            ts = time.time()
            
            np_state = np.array(state.obs)
            observation = np.array([np_state[0], np_state[1], np_state[5], np_state[6], np_state[9]])
            prob  = agent.act(observation, eps) 
            action_indicator = np.argmax(prob)
            
            # discretize env
            if action_indicator == 0:
                action = np.array([.1])
            else:
                action = np.array([-.1])

            #print(action_indicator)

            #print(eps, eps_decay, min_eps)
            eps = max(eps*eps_decay, min_eps)

            
            old_observation = observation

            #observation, reward, done, _ = env.step(torch.tensor(np.array(action[None,:])))
            #observation = observation.numpy()[0]
            #state = env.step(state, action)
            state = jax.jit(env.step)(state, action)
            yield image.render_array(env.sys, state.qp, width=500, height=500, ssaa=1)
            np_state = np.array(state.obs)
            observation = np.array([np_state[0], np_state[1], np_state[5], np_state[6], np_state[9]])
            reward = state.reward
            done = state.done
            
            #actor.remember(old_observation, action, reward, observation, done)
            episode_rewards.append(reward)
            ep_rewards += reward
            te = time.time()
            #feedback_value = 0 # uncomment for sparse feedback
            time.sleep(e) # Delay to make the game seeable
            feedback = ""
            all_feedback = np.genfromtxt(f"tamer_feedback_data/participant_feed_{uuid}.csv", delimiter=',')

            if len(all_feedback) > 0:
                if len(all_feedback) == 2 and np.size(all_feedback) == 2:
                    latest_feedback = all_feedback
                else:
                    latest_feedback = all_feedback[len(all_feedback)-1] 
                    feedback = latest_feedback[0]
                    tf_old = copy.copy(tf) 
                    tf = latest_feedback[1]
                    if tf != tf_old:
                        if feedback == 1:
                            feedback_value = 1
                            loss = action_queue.push_to_buffer(agent, tf, feedback_value, interval_min=interval_min, interval_max=interval_max)
                            with open(state_filename, 'a', newline='') as csvfile:  
                                csvwriter = csv.writer(csvfile)  
                                csvwriter.writerow([observation, tf]) 
                                csvfile.close()    
                        elif feedback == -1:
                            feedback_value = -1
                            loss = action_queue.push_to_buffer(agent, tf, feedback_value, interval_min=interval_min, interval_max=interval_max)
                            with open(state_filename, 'a', newline='') as csvfile:  
                                csvwriter = csv.writer(csvfile)  
                                csvwriter.writerow([observation, tf]) 
                                csvfile.close()    
                        #else: 
                            #feedback_value = 0 
                            #loss = action_queue.push_to_buffer(agent, tf, feedback_value, interval_min=interval_min, interval_max=interval_max)
            #print(tf, ts, tf-ts)
            action_queue.enqeue([observation, action, ts, te, tf, feedback_value, old_observation, done])
            
            if feedback_value != 0:
                agent.step(old_observation, action_indicator,feedback_value, observation, done)
            else:
                agent.step_without_storage()

            feedback_value=0
            # if feedback_value != 0:
            #     #tf = stopwatch.duration
            #     # [observation, action, ts, te, tf, feedback_value, mu, sigma, old_observation, done]
            #     #loss = action_queue.push_to_buffer_and_learn(agent, actor, tf, feedback_value, interval_min=interval_min, interval_max=interval_max)
            #     agent.remember(old_observation, action, feedback_value, observation, done)
            #     #print(loss, "feedback loss")


            if stopwatch.duration > 60:
                done = True

            if done:
                with open(rew_filename, 'a', newline='') as csvfile:  
                    csvwriter = csv.writer(csvfile)  
                    csvwriter.writerow([episode_num, ep_rewards, end_f, time.time()]) 
                    csvfile.close()                                                         
                #print(ep_rewards)
                #print("Episode:", str(episode_num))
                rewards.append(ep_rewards)
                ep_rewards = 0
                episode_rewards = []
                episode_num += 1
                feedback_value = 0
                #action_queue.clear()

                #if len(rewards) > 0:
                    #if rewards[len(rewards)-1] >= 400:
                        #true_done = True
                break

            if time.time() > timeout and done:
                true_done = True
                break         
            
    finished = True
    #env.close()
    del env, agent, action_queue
    img = cv2.imread("cair_outro_slide_image.jpg")

    cv2.putText(img,str(uuid), (270, 550), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)
    
    arr = np.array(img)
    yield arr
    return True


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2659)
