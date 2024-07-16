lrs=("5e-4" "1e-3" "5e-3" "1e-2")

epochs=("100" "200" "500" "1000")


while true
do 
    for lr in "${lrs[@]}"
    do
        for epoch in "${epochs[@]}"
        do
            python train_v2.py --lr "$lr" --epochs "$epoch"
        done
    done
done
