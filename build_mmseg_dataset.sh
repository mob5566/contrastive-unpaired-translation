#!/bin/sh

TRAN_SCRIPT=$1
RESULT_DIR=$2
TRAIN_LABEL_DIR=$3
TEST_LABEL_DIR=$4
TARGET_DIR=$5

for experiment in $(ls "$RESULT_DIR") ; do
	for phase_epoch in $(ls "$RESULT_DIR/$experiment") ; do
		phase=$(echo $phase_epoch | cut -d_ -f1)
		epoch=$(echo $phase_epoch | cut -d_ -f2)
		target_ds_base="$TARGET_DIR/synhm.$experiment.ep$epoch/dataset.mmseg.png"

		if [ "$phase" = "train" ] ; then
			target_phase="train"
			source_img_dir="$RESULT_DIR/$experiment/$phase_epoch/images/fake_B"
			source_label_base="$TRAIN_LABEL_DIR/dataset.mmseg.png/ann_dir"
		else
			target_phase="val"
			source_img_dir="$TEST_LABEL_DIR/dataset.mmseg.png/img_dir/val"
			source_label_base="$TEST_LABEL_DIR/dataset.mmseg.png/ann_dir"
		fi

		"$TRAN_SCRIPT" --mode dir --transform hardlink --src MM --dest MM \
			"$source_img_dir" \
			"$target_ds_base/img_dir/$target_phase"

		"$TRAN_SCRIPT" --mode dir --transform hardlink --src MM --dest MM \
			"$source_label_base/$target_phase" \
			"$target_ds_base/ann_dir/$target_phase"
	done
done



