# ABC



## Dependencies

## Train model
The code `source_free.py` can be used for our algorithm.
```
python3 source_free.py --total_epoch 10 --target_type $DA_TYPE --dataset $DATASET --source $SRC --target $TAR --balance $BALANCE_PARAMETER --lr $LR '
                                                               '--lr_scale 0.1 --max_k 100 --KK $KK --covariance_prior 0.001 --score cos --classifier 
```
You can also check the make files in the repo to see how to run different tasks.
## License
This code is released under the MIT License.
