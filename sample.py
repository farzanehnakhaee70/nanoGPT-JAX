from dataclasses import dataclass
from functools import partial
import pickle

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.training import train_state
from flax import serialization

import optax
import time


@dataclass
class Config():
    seed = 42
    num_iterations = 20000
    batch_size = 1
    block_size = 1024
    learning_rate = 1e-4
    embed_size = 768
    num_heads = 1
    head_size = 768
    num_layers = 12
    dropout = 0.0

config = Config()
vocab_size = 50257

class LayerNorm(nn.Module):
    epsilon: float = 1e-6
    reduction_axes = -1

    @nn.compact
    def __call__(self, x):
        """Applies layer normalization on the input."""
        # compute statistics
        mean2 = jnp.mean(jax.lax.square(x), self.reduction_axes, keepdims=True)
        mean = jnp.mean(x, self.reduction_axes, keepdims=True)
        var = jnp.maximum(0., mean2 - jax.lax.square(mean))

        # compute normalized inputs
        x_norm = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        return x_norm * self.param("scale", nn.initializers.ones, x.shape[-1]) + self.param("bias", nn.initializers.zeros, x.shape[-1])

class Attention(nn.Module):
    head_size: int

    @nn.compact
    def __call__(self, x, training: bool):
        key = nn.Dense(self.head_size, use_bias=False)(x)
        query = nn.Dense(self.head_size, use_bias=False)(x)
        value = nn.Dense(self.head_size, use_bias=False)(x)
        
        tril = jnp.tril(jnp.ones((x.shape[-2], x.shape[-2])))
        attention_weights = nn.softmax(jnp.where(tril == 0, -jnp.inf, query @ jnp.transpose(key, axes=(0, 2, 1))), axis=-1)
        attention_weights = nn.Dropout(config.dropout)(attention_weights, deterministic=not training)
        return attention_weights @ value

class MultiHeadAttention(nn.Module):
    num_heads: int
    head_size: int

    @nn.compact
    def __call__(self, x, training: bool):
        x = jnp.concatenate([Attention(self.head_size)(x, training) for _ in range(self.num_heads)], axis=-1)
        return nn.Dropout(config.dropout)(nn.Dense(self.num_heads*self.head_size)(x), deterministic=not training)

class FeedFoward(nn.Module):

    @nn.compact
    def __call__(self, x, training: bool):
        return nn.Dropout(config.dropout)(nn.Dense(config.embed_size)(nn.relu(nn.Dense(4*config.embed_size)(x))), deterministic=not training)

class Block(nn.Module):
    num_heads: int
    head_size: int

    @nn.compact
    def __call__(self, x, training: bool):
        x = x + MultiHeadAttention(self.num_heads, self.head_size)(LayerNorm()(x), training)
        return x + FeedFoward()(LayerNorm()(x), training)

class Model(nn.Module):
    num_layers: int
    num_heads: int
    head_size: int

    @nn.compact
    def __call__(self, x, training: bool):
        B, T = x.shape
        x = nn.Embed(num_embeddings=vocab_size, features=config.embed_size)(x) + \
            nn.Embed(num_embeddings=config.block_size, features=config.embed_size)(jnp.arange(T))
        for _ in range(self.num_layers):
            x = Block(self.num_heads, self.head_size)(x, training)
        x = nn.LayerNorm()(x)
        return nn.Dense(vocab_size)(x)

    def generate(self, random_key, params, context, length=50):
        for _ in range(length):
            logits = self.apply(params, context[:, -config.block_size:], training=False)
            random_key, random_subkey = jax.random.split(random_key)
            new_token = jax.random.categorical(random_subkey, logits[:, -1, :], axis=-1, shape=(1, 1))
            context = jnp.concatenate([context, new_token], axis=1)
        return context

    @partial(jax.jit, static_argnames=("self", "length"))
    def generate_jit(self, random_key, params, length):
        def scan_generate(carry, x):
            key, context = carry
            logits = self.apply(params, context, training=False)
            random_key, random_subkey = jax.random.split(key)
            new_token = jax.random.categorical(random_subkey, logits[:, -1, :], axis=-1, shape=(1, 1))
            context = jnp.concatenate([context[:, 1:], new_token], axis=1)
            return (random_key, context), new_token
        
        _, new_tokens = jax.lax.scan(
            scan_generate,
            (random_key, jnp.zeros((1, config.block_size), dtype=jnp.int32)),
            (),
            length=length,
        )
        return new_tokens

random_key = jax.random.PRNGKey(config.seed)
random_key, random_subkey = jax.random.split(random_key)

# Let's now generate some text
model = Model(num_layers=config.num_layers, num_heads=config.num_heads, head_size=config.head_size)
params = model.init(
    random_key, jnp.ones((config.batch_size, config.block_size), dtype=jnp.int32), training=False
)

for i in range(10):
    t1 = time.time()
    text = model.generate_jit(random_key, params, 500)[:, 0, 0].tolist()
    t2 = time.time()
    print(t2 - t1)
