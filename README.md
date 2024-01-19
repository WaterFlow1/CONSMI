# CONSMI

## checkpoint 
https://drive.google.com/file/d/128GZoN2O9rfVGR24q1vPgkAOtEgTKEry/view?usp=drive_link

## dataset
### zinc
https://drive.google.com/file/d/1fnUAsJ78kIMpXMxIEYAeC46plSM2DAzu/view?usp=drive_link
### celegan
https://drive.google.com/file/d/1D7ha2lUDacA-ST4dFu01i_bbHmKfof5E/view?usp=drive_link
### davis
https://drive.google.com/file/d/1KdbkxOtL1J8AIKKZSepDFzuFP6SUcvJy/view?usp=drive_link

### molgpt
The code related to molgpt is available in the github:

https://github.com/devalab/molgpt.git

## How to train

`python3 pretrain_con.py -vocab True`

## How to use

Ensure vocabulary consistency.

Add this code:

```python
pre_model = TextEmbedding(src_vocab_size=95)
pre_model.load_state_dict(torch.load('./results/best_steps_checkpoint.pt'))
model.src_emb.load_state_dict(pre_model.tok_emb.state_dict())
model.src_emb.requires_grad_(False)
```


