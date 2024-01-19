# python main_train_dncnn.py

# python main_train_drunet.py


# LOSS_FORM="1 2 3 4 5 6 7 8 9"

# for loss_form in $LOSS_FORM
# do
#     python main_train_dncnn.py --G_loss_form=$loss_form --gpu_ids="[0]"
# done



# python main_train_dncnn.py --G_loss_form=0 --gpu_ids="[0]" &
# python main_train_dncnn.py --G_loss_form=1 --gpu_ids="[1]" &
# python main_train_dncnn.py --G_loss_form=2 --gpu_ids="[2]" &
# python main_train_dncnn.py --G_loss_form=3 --gpu_ids="[3]" &
# python main_train_dncnn.py --G_loss_form=4 --gpu_ids="[]"
# python main_train_dncnn.py --G_loss_form=5 --gpu_ids="[]"
# python main_train_dncnn.py --G_loss_form=6 --gpu_ids="[]"

# python main_train_dncnn.py --G_loss_form=6 --suffix=0 --gpu_ids="[0]" \
#             --opt=options/train_dncnn.json


# python main_train_drunet.py --G_loss_form=0 --suffix=123 --gpu_ids="[0]" \
#             --opt=options/train_drunet.json




# --------------------UNet-------------------- #
# python main_train_unet.py --G_loss_form=0 --suffix=basic --gpu_ids="[0,1,2,3]" \
#             --opt=options/train_unet.json

python main_train_unet.py --G_loss_form=6 --suffix=heron --gpu_ids="[0,1,2,3]" \
            --opt=options/train_unet.json