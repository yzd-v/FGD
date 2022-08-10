_base_ = [
    '../../yolox/yolox_m_8x8_300e_coco.py'
]
# model settings
find_unused_parameters=True
temp=0.5
alpha_fgd=0.002
beta_fgd=0.001
gamma_fgd=0.001
lambda_fgd=0.00001
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
    init_student = False,
    yolox = True,
    distill_cfg = [ dict(student_module = 'neck.out_convs.0',
                         teacher_module = 'neck.out_convs.0',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_0',
                                       student_channels = 192,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.out_convs.1',
                         teacher_module = 'neck.out_convs.1',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_1',
                                       student_channels = 192,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.out_convs.2',
                         teacher_module = 'neck.out_convs.2',
                         output_hook = True,
                         methods=[dict(type='FeatureLoss',
                                       name='loss_fgd_fpn_2',
                                       student_channels = 192,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),

                   ]
    )


student_cfg = 'configs/yolox/yolox_m_8x8_300e_coco.py'
teacher_cfg = 'configs/yolox/yolox_l_8x8_300e_coco.py'
