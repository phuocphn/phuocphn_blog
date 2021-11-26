**Mistake #1**

Every time you shuffle dataset, you need to assign it again, it will return the new shuffled dataset, and `shuffle()` is not a in-place operation.

```python
dataset = RCLDataset()

# wrong approach !
dataset.shuffle()

# correct approach
dataset = dataset.shuffle()
```

#### **Mistake #2**

Can not load the pre-trained model with Pytorch Lightning

```python

# wrong
model = MyModel(*args)
model.load_from_checkpoint(ckpt)

# correct approach
model = MyModel(*args).load_from_checkpoint(ckpt)
```
