# Transformer Decoder 

## Objective 
The existing transformer libraries doesn't have KV cache and wasn't flexible to use as per the need. This will solve the problems those implementations have.
This is built entirely on C++ with the use of Libtorch library (Pytorch with C++ API) faster and flexible 

## Working
The Transfomer takes a struct "TransformerConfig" which includes the attributes of the individual components of entire model.