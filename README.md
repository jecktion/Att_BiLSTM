# Relation Classification with Attention-Bi-LSTM

### [中文Blog](https://blog.csdn.net/Kaiyuan_sjtu/article/details/89961647)


Original paper  [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification（Zhou/ACL2016）](https://www.aclweb.org/anthology/P16-2034) with overall architecture listed below.

![](https://img-blog.csdnimg.cn/20190508165045537.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ==,size_16,color_FFFFFF,t_70)


## Requrements

* Python (>=3.5)

* TensorFlow (>=r1.0)

## Datasets
- SemEval 2010 Task 8 with 19 relations.The full introduction can be found [here](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview).


## Usage

### For Training:

```
#python3 train.py 
```

### For final evaluate
 
```
#python3 eval.py 
```

## To do
- finetune the paras to get better performance
- other todos are in the code
- any suggestions, welcome to issue

