import numpy as np
import random
from collections import deque
import torch


class ReplayBuffer:
    """
    Improved Replay Buffer for TD3 with enhanced sampling and memory management
    """
    def __init__(self, state_dim, action_dim, max_size=1e6, device='cuda'):
        # ... keep all your existing __init__ code ...
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        
        # Initialize storage arrays (keep existing)
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)
        
        # Device for tensor operations
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # ADD THIS: GPU tensors for fast sampling
        self.gpu_state = None
        self.gpu_action = None
        self.gpu_next_state = None
        self.gpu_reward = None
        self.gpu_done = None
        self.gpu_sync_needed = True
        
        # ... keep all your existing priority/statistics code ...
        self.priorities = np.zeros(self.max_size, dtype=np.float32)
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.max_priority = 1.0
        self.total_added = 0
        self.collision_buffer = deque(maxlen=10000)
        self.success_buffer = deque(maxlen=10000)
        
    def add(self, state, action, next_state, reward, done):
        """Store experience in buffer; convert tensors if needed"""
        # Convert torch tensors (especially from CUDA) to numpy
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.detach().cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.detach().cpu().numpy()

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        
        # Assign priority (higher for important experiences)
        priority = self.max_priority
        if abs(reward) > 50:
            priority = self.max_priority * 1.5
        elif abs(reward) > 20:
            priority = self.max_priority * 1.2
        
        self.priorities[self.ptr] = priority
        
        # Store special experiences in separate buffers
        if reward > 150:  # Success
            self.success_buffer.append({
                'state': state.copy(),
                'action': action.copy(),
                'next_state': next_state.copy(),
                'reward': reward,
                'done': done
            })
        elif reward < -150:  # Collision
            self.collision_buffer.append({
                'state': state.copy(),
                'action': action.copy(),
                'next_state': next_state.copy(),
                'reward': reward,
                'done': done
            })
        
        # Update pointers
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.total_added += 1
        
        # Mark GPU sync needed
        self.gpu_sync_needed = True

    def _sync_to_gpu(self):
        """Sync numpy arrays to GPU tensors for fast sampling"""
        if self.size == 0:
            # Initialize empty tensors if buffer is empty
            self.gpu_state = torch.empty((0, self.state.shape[1]), device=self.device)
            self.gpu_action = torch.empty((0, self.action.shape[1]), device=self.device)
            self.gpu_next_state = torch.empty((0, self.next_state.shape[1]), device=self.device)
            self.gpu_reward = torch.empty((0, 1), device=self.device)
            self.gpu_done = torch.empty((0, 1), device=self.device)
            self.gpu_sync_needed = False
            return

        if not self.gpu_sync_needed:
            return
            
        current_size = self.size
        self.gpu_state = torch.tensor(self.state[:current_size], dtype=torch.float32, device=self.device)
        self.gpu_action = torch.tensor(self.action[:current_size], dtype=torch.float32, device=self.device)
        self.gpu_next_state = torch.tensor(self.next_state[:current_size], dtype=torch.float32, device=self.device)
        self.gpu_reward = torch.tensor(self.reward[:current_size], dtype=torch.float32, device=self.device)
        self.gpu_done = torch.tensor(self.done[:current_size], dtype=torch.float32, device=self.device)
        
        self.gpu_sync_needed = False
        
    def sample(self, batch_size, use_gpu=True):
        """
        Enhanced sample method with GPU option
        use_gpu=True: Returns GPU tensors (fast)
        use_gpu=False: Returns numpy arrays (your original method)
        """
        if use_gpu:
            return self._sample_gpu(batch_size)
        else:
            return self._sample_cpu(batch_size)
    
    def _sample_gpu(self, batch_size):
        """GPU-optimized sampling that returns tensors"""
        # Sync to GPU if needed
        self._sync_to_gpu()

        if self.size == 0:
            # Return empty tensors if buffer is empty
            return (
                torch.empty((0, self.state.shape[1]), device=self.device),
                torch.empty((0, self.action.shape[1]), device=self.device),
                torch.empty((0, self.next_state.shape[1]), device=self.device),
                torch.empty((0, 1), device=self.device),
                torch.empty((0, 1), device=self.device),
            )

        # Get indices using enhanced sampling
        indices_np = self._enhanced_sample(batch_size)
        indices = torch.tensor(indices_np, dtype=torch.long, device=self.device)

        # Make sure indices are within bounds
        indices = indices.clamp(0, self.size - 1)

        return (
            self.gpu_state[indices],
            self.gpu_action[indices], 
            self.gpu_next_state[indices],
            self.gpu_reward[indices],
            self.gpu_done[indices]
        )
    
    def _sample_cpu(self, batch_size):
        """Your original CPU sampling method"""
        if self.size < batch_size:
            indices = np.arange(self.size)
            batch_size = self.size
        else:
            indices = self._enhanced_sample(batch_size)
        
        return (
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            self.reward[indices],
            self.done[indices]
        )
    
    def _enhanced_sample(self, batch_size):
        """
        Enhanced sampling with priority and special experience replay
        """
        # Calculate sampling distribution
        if self.size == self.max_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]
        
        # Apply priority weighting
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample most indices based on priority
        main_batch_size = int(batch_size * 0.7)  # 70% priority sampling
        special_batch_size = batch_size - main_batch_size
        
        # Priority-based sampling
        main_indices = np.random.choice(
            self.size, main_batch_size, p=probs, replace=True
        )
        
        # Sample special experiences (success/collision) if available
        special_indices = []
        if len(self.success_buffer) > 0 and len(self.collision_buffer) > 0:
            # Sample from both success and collision buffers
            n_success = special_batch_size // 2
            n_collision = special_batch_size - n_success
            
            success_samples = random.sample(
                self.success_buffer, min(n_success, len(self.success_buffer))
            )
            collision_samples = random.sample(
                self.collision_buffer, min(n_collision, len(self.collision_buffer))
            )
            
            # Add special samples to buffer temporarily and get their indices
            special_indices = self._add_temporary_samples(
                success_samples + collision_samples
            )
        
        # Combine indices - Fixed the bug here
        if len(special_indices) > 0:  # Check length instead of truthiness
            indices = np.concatenate([main_indices, special_indices])
        else:
            # If no special samples, just sample more from main buffer
            indices = np.random.choice(
                self.size, batch_size, p=probs, replace=True
            )
        
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return indices
    
    def _add_temporary_samples(self, samples):
        """
        Temporarily add special samples for current batch (helper method)
        """
        indices = []
        for sample in samples:
            if self.size < self.max_size:
                # If buffer not full, add permanently
                self.state[self.size] = sample['state']
                self.action[self.size] = sample['action']
                self.next_state[self.size] = sample['next_state']
                self.reward[self.size] = sample['reward']
                self.done[self.size] = sample['done']
                self.priorities[self.size] = self.max_priority * 1.5
                
                indices.append(self.size)
                self.size += 1
            else:
                # If buffer full, replace random entry temporarily
                idx = np.random.randint(0, self.size)
                indices.append(idx)

        self.gpu_sync_needed = True
        
        return np.array(indices)
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors (for prioritized replay)
        """
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6  # Small epsilon to avoid zero priority
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def sample_uniform(self, batch_size):
        """
        Sample uniformly from the buffer (standard replay)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            self.reward[indices],
            self.done[indices]
        )
    
    def get_statistics(self):
        """
        Get buffer statistics for monitoring
        """
        return {
            'size': self.size,
            'total_added': self.total_added,
            'success_samples': len(self.success_buffer),
            'collision_samples': len(self.collision_buffer),
            'avg_reward': np.mean(self.reward[:self.size]) if self.size > 0 else 0,
            'reward_std': np.std(self.reward[:self.size]) if self.size > 0 else 0,
            'max_priority': self.max_priority,
            'beta': self.beta
        }
    
    def save(self, filename):
        """
        Save replay buffer to disk
        """
        np.savez_compressed(
            filename,
            state=self.state[:self.size],
            action=self.action[:self.size],
            next_state=self.next_state[:self.size],
            reward=self.reward[:self.size],
            done=self.done[:self.size],
            priorities=self.priorities[:self.size],
            size=self.size,
            ptr=self.ptr,
            total_added=self.total_added
        )
        print(f"Replay buffer saved to {filename}")
    
    def load(self, filename):
        """
        Load replay buffer from disk
        """
        data = np.load(filename)
        
        size = data['size']
        self.state[:size] = data['state']
        self.action[:size] = data['action']
        self.next_state[:size] = data['next_state']
        self.reward[:size] = data['reward']
        self.done[:size] = data['done']
        self.priorities[:size] = data['priorities']
        
        self.size = size
        self.ptr = data['ptr']
        self.total_added = data['total_added']
        self.max_priority = np.max(self.priorities[:size]) if size > 0 else 1.0
        
        print(f"Replay buffer loaded from {filename}")
        print(f"Buffer size: {self.size}, Total experiences: {self.total_added}")
    
    def clear(self):
        """
        Clear the replay buffer
        """
        self.ptr = 0
        self.size = 0
        self.total_added = 0
        self.priorities.fill(0)
        self.max_priority = 1.0
        self.success_buffer.clear()
        self.collision_buffer.clear()
        print("Replay buffer cleared")


class HindsightExperienceReplay(ReplayBuffer):
    """
    Enhanced replay buffer with Hindsight Experience Replay (HER) for goal-conditioned tasks
    """
    def __init__(self, state_dim, action_dim, max_size=1e6, her_fraction=0.8, device='cuda'):
        super().__init__(state_dim, action_dim, max_size, device)
        self.her_fraction = her_fraction  # Fraction of HER samples in each batch
        self.episode_buffer = []  # Store current episode
        
    def add_episode_step(self, state, action, next_state, reward, done, goal_achieved=False):
        """
        Add step to current episode buffer
        """
        self.episode_buffer.append({
            'state': state.copy(),
            'action': action.copy(),
            'next_state': next_state.copy(),
            'reward': reward,
            'done': done,
            'goal_achieved': goal_achieved
        })
        
        # If episode is done, process HER and add to main buffer
        if done:
            self._process_episode_with_her()
            self.episode_buffer = []
    
    def _process_episode_with_her(self):
        """
        Process episode with Hindsight Experience Replay
        """
        if not self.episode_buffer:
            return
        
        # Add original episode to buffer
        for step in self.episode_buffer:
            self.add(
                step['state'], step['action'], step['next_state'],
                step['reward'], step['done']
            )
        
        # Generate HER samples
        episode_length = len(self.episode_buffer)
        if episode_length > 1:
            # For each step, create HER samples with future states as goals
            for t in range(episode_length - 1):
                # Sample random future step as new goal
                future_step = np.random.randint(t + 1, episode_length)
                
                # Create new experience with hindsight goal
                her_reward = self._compute_her_reward(
                    self.episode_buffer[t]['next_state'],
                    self.episode_buffer[future_step]['state']
                )
                
                # Add HER experience
                self.add(
                    self.episode_buffer[t]['state'],
                    self.episode_buffer[t]['action'],
                    self.episode_buffer[t]['next_state'],
                    her_reward,
                    self.episode_buffer[t]['done']
                )
    
    def _compute_her_reward(self, achieved_state, goal_state):
        """
        Compute reward for HER based on achieved state and hindsight goal
        """
        # Extract position from state (assuming first 2 elements are x, y)
        achieved_pos = achieved_state[:2]
        goal_pos = goal_state[:2]
        
        distance = np.linalg.norm(achieved_pos - goal_pos)
        
        # Dense reward based on distance
        if distance < 0.3:  # Goal reached
            return 200.0
        else:
            return -distance * 10.0  # Distance penalty


# Example usage and testing
if __name__ == "__main__":
    # Test the replay buffer
    state_dim = 27  # 20 (sensors) + 7 (robot state)
    action_dim = 2
    
    # Create buffer
    buffer = ReplayBuffer(state_dim, action_dim, max_size=10000)
    
    # Add some dummy experiences
    for i in range(1000):
        state = np.random.randn(state_dim)
        action = np.random.uniform([0, -1], [1, 1])
        next_state = np.random.randn(state_dim)
        reward = np.random.uniform(-100, 100)
        done = np.random.choice([0, 1], p=[0.95, 0.05])
        
        buffer.add(state, action, next_state, reward, done)
    
    # Sample batch
    batch = buffer.sample(256)
    print("Batch shapes:")
    for i, tensor in enumerate(batch):
        print(f"  {i}: {tensor.shape}")
    
    # Print statistics
    stats = buffer.get_statistics()
    print("\nBuffer Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test save/load
    buffer.save("test_buffer.npz")
    new_buffer = ReplayBuffer(state_dim, action_dim)
    new_buffer.load("test_buffer.npz")
    
    print("\nReplay buffer implementation complete!")