from basic import *

from attention import SAMMAFB

class three_branch_bb(nn.Module):
    def __init__(self, args):
        super(three_branch_bb, self).__init__()
        self.args = args
 
        self.cbam_128 = SAMMAFB(128)
        self.cbam_256 = SAMMAFB(256)
        self.cbam_512 = SAMMAFB(512)
        self.cbam_1024 = SAMMAFB(1024)                

        self.cbam_192 = SAMMAFB(192)
        self.cbam_448 = SAMMAFB(448)
        self.cbam_960 = SAMMAFB(960)
        self.cbam_1984 = SAMMAFB(1984)                



        self.rgb_conv_init = convbnrelu(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.rgb_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2)
        self.rgb_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1)
        self.rgb_encoder_layer3 = BasicBlockGeo(inplanes=64, planes=128, stride=2)
        self.rgb_encoder_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1)
        self.rgb_encoder_layer5 = BasicBlockGeo(inplanes=128, planes=256, stride=2)
        self.rgb_encoder_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1)
        self.rgb_encoder_layer7 = BasicBlockGeo(inplanes=256, planes=512, stride=2)
        self.rgb_encoder_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1)
        self.rgb_encoder_layer9 = BasicBlockGeo(inplanes=512, planes=1024, stride=2)
        self.rgb_encoder_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1)

        self.rgb_decoder_layer8 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer6 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer4 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer2 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_output = deconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)

        #Segmentation Encoder

        self.segment_conv_init = convbnrelu(in_channels=5, out_channels=32, kernel_size=5, stride=1, padding=2)
        
        self.segment_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2)
        self.segment_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1)
        self.segment_encoder_layer3 = BasicBlockGeo(inplanes=128, planes=128, stride=2)
        self.segment_encoder_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1)
        self.segment_encoder_layer5 = BasicBlockGeo(inplanes=256, planes=256, stride=2)
        self.segment_encoder_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1)
        self.segment_encoder_layer7 = BasicBlockGeo(inplanes=512, planes=512, stride=2)
        self.segment_encoder_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1)
        self.segment_encoder_layer9 = BasicBlockGeo(inplanes=1024, planes=1024, stride=2)
        self.segment_encoder_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1)

        self.segment_decoder_layer1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.segment_decoder_layer2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.segment_decoder_layer3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.segment_decoder_layer4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.segment_decoder_layer5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.segment_decoder_layer6 = deconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)




        # depth encoder
        self.depth_conv_init = convbnrelu(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.depth_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2)
        self.depth_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1)
        
        self.depth_layer3 = BasicBlockGeo(inplanes=192, planes=192, stride=2)
        self.depth_layer4 = BasicBlockGeo(inplanes=192, planes=192, stride=1)
        
        self.depth_layer5 = BasicBlockGeo(inplanes=448, planes=448, stride=2)
        self.depth_layer6 = BasicBlockGeo(inplanes=448, planes=448, stride=1)
        
        self.depth_layer7 = BasicBlockGeo(inplanes=960, planes=960, stride=2)
        self.depth_layer8 = BasicBlockGeo(inplanes=960, planes=960, stride=1)
        
        self.depth_layer9 = BasicBlockGeo(inplanes=1984, planes=1984, stride=2)
        self.depth_layer10 = BasicBlockGeo(inplanes=1984, planes=1984, stride=1)


        # decoder
        self.decoder_layer0 = deconvbnrelu(in_channels=1984, out_channels=1024, kernel_size=1,stride=1, padding=0, output_padding=0)
        self.decoder_layer1 = deconvbnrelu(in_channels=1024, out_channels=960, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer2 = deconvbnrelu(in_channels=960, out_channels=448, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer3 = deconvbnrelu(in_channels=448, out_channels=192, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer4 = deconvbnrelu(in_channels=192, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)

        self.decoder_layer6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)



        weights_init(self)

    def forward(self, input):
       
        rgb = input['rgb']
        d = input['d']
        semantic = input['semantic']
        

        rgb_feature = self.rgb_conv_init(torch.cat((rgb, d), dim=1))
        rgb_feature1 = self.rgb_encoder_layer1(rgb_feature) 
        rgb_feature2 = self.rgb_encoder_layer2(rgb_feature1) 
        rgb_feature3 = self.rgb_encoder_layer3(rgb_feature2) 
        rgb_feature4 = self.rgb_encoder_layer4(rgb_feature3) 
        rgb_feature5 = self.rgb_encoder_layer5(rgb_feature4) 
        rgb_feature6 = self.rgb_encoder_layer6(rgb_feature5) 
        rgb_feature7 = self.rgb_encoder_layer7(rgb_feature6) 
        rgb_feature8 = self.rgb_encoder_layer8(rgb_feature7) 
        rgb_feature9 = self.rgb_encoder_layer9(rgb_feature8) 
        rgb_feature10 = self.rgb_encoder_layer10(rgb_feature9) 

        rgb_feature_decoder8 = self.rgb_decoder_layer8(rgb_feature10)
        rgb_feature8_plus = rgb_feature_decoder8 + rgb_feature8

        rgb_feature_decoder6 = self.rgb_decoder_layer6(rgb_feature8_plus)
        rgb_feature6_plus = rgb_feature_decoder6 + rgb_feature6

        rgb_feature_decoder4 = self.rgb_decoder_layer4(rgb_feature6_plus)
        rgb_feature4_plus = rgb_feature_decoder4 + rgb_feature4

        rgb_feature_decoder2 = self.rgb_decoder_layer2(rgb_feature4_plus)
        rgb_feature2_plus = rgb_feature_decoder2 + rgb_feature2   # b 32 176 608

        rgb_feature_decoder0 = self.rgb_decoder_layer0(rgb_feature2_plus)
        rgb_feature0_plus = rgb_feature_decoder0 + rgb_feature

        rgb_output = self.rgb_decoder_output(rgb_feature0_plus)
        rgb_depth = rgb_output[:, 0:1, :, :]
        rgb_conf = rgb_output[:, 1:2, :, :]



        semantic_feature = self.segment_conv_init(torch.cat((semantic, d, rgb_depth), dim=1))
        semantic_feature1 = self.segment_encoder_layer1(semantic_feature) 
        semantic_feature2 = self.segment_encoder_layer2(semantic_feature1)

        semantic_feature2_plus = torch.cat([rgb_feature2_plus, semantic_feature2], 1)

        semantic_feature2_plus = self.cbam_128(semantic_feature2_plus)

        semantic_feature3 = self.segment_encoder_layer3(semantic_feature2_plus) 
        semantic_feature4 = self.segment_encoder_layer4(semantic_feature3) 

        semantic_feature4_plus = torch.cat([rgb_feature4_plus, semantic_feature4], 1)

        semantic_feature4_plus = self.cbam_256(semantic_feature4_plus)

        semantic_feature5 = self.segment_encoder_layer5(semantic_feature4_plus)
        semantic_feature6 = self.segment_encoder_layer6(semantic_feature5) 

        semantic_feature6_plus = torch.cat([rgb_feature6_plus, semantic_feature6], 1)

        semantic_feature6_plus = self.cbam_512(semantic_feature6_plus)

        semantic_feature7 = self.segment_encoder_layer7(semantic_feature6_plus) 
        semantic_feature8 = self.segment_encoder_layer8(semantic_feature7) 

        semantic_feature8_plus = torch.cat([rgb_feature8_plus, semantic_feature8], 1)

        semantic_feature8_plus = self.cbam_1024(semantic_feature8_plus)

        semantic_feature9 = self.segment_encoder_layer9(semantic_feature8_plus) 
        semantic_feature10 = self.segment_encoder_layer10(semantic_feature9) 


        semantic_fusion1 = rgb_feature10 + semantic_feature10
        semantic_feature_decoder1 = self.segment_decoder_layer1(semantic_fusion1)

        semantic_fusion2 = semantic_feature_decoder1 + semantic_feature8
        semantic_feature_decoder2 = self.segment_decoder_layer2(semantic_fusion2)

        semantic_fusion3 = semantic_feature_decoder2 + semantic_feature6
        semantic_feature_decoder3 = self.segment_decoder_layer3(semantic_fusion3)

        semantic_fusion4 = semantic_feature_decoder3 + semantic_feature4
        semantic_feature_decoder4 = self.segment_decoder_layer4(semantic_fusion4)

        semantic_fusion5 = semantic_feature_decoder4 + semantic_feature2
        semantic_feature_decoder5 = self.segment_decoder_layer5(semantic_fusion5)

        semantic_depth_output = self.segment_decoder_layer6(semantic_feature_decoder5)
        semantic_depth, semantic_conf = torch.chunk(semantic_depth_output, 2, dim=1)


        sparsed_feature = self.depth_conv_init(torch.cat((d, rgb_depth,semantic_depth), dim=1))
        sparsed_feature1 = self.depth_layer1(sparsed_feature)
        sparsed_feature2 = self.depth_layer2(sparsed_feature1) 

        sparsed_feature2_plus = torch.cat([rgb_feature2_plus, semantic_feature_decoder4, sparsed_feature2], 1)

        sparsed_feature2_plus = self.cbam_192(sparsed_feature2_plus)
        
        sparsed_feature3 = self.depth_layer3(sparsed_feature2_plus) 
        sparsed_feature4 = self.depth_layer4(sparsed_feature3) 

        sparsed_feature4_plus = torch.cat([rgb_feature4_plus, semantic_feature_decoder3, sparsed_feature4], 1)
        
        sparsed_feature4_plus = self.cbam_448 (sparsed_feature4_plus)

        sparsed_feature5 = self.depth_layer5(sparsed_feature4_plus) 
        sparsed_feature6 = self.depth_layer6(sparsed_feature5) 

        sparsed_feature6_plus = torch.cat([rgb_feature6_plus, semantic_feature_decoder2, sparsed_feature6], 1)

        sparsed_feature6_plus = self.cbam_960(sparsed_feature6_plus)

        sparsed_feature7 = self.depth_layer7(sparsed_feature6_plus) 
        sparsed_feature8 = self.depth_layer8(sparsed_feature7) 

        sparsed_feature8_plus = torch.cat([rgb_feature8_plus, semantic_feature_decoder1, sparsed_feature8], 1)
        
        sparsed_feature8_plus = self.cbam_1984(sparsed_feature8_plus)

        sparsed_feature9 = self.depth_layer9(sparsed_feature8_plus) 
        sparsed_feature10 = self.depth_layer10(sparsed_feature9) 


        decoder_feature0 = self.decoder_layer0(sparsed_feature10)

        fusion1 = semantic_fusion1 + decoder_feature0
        decoder_feature1 = self.decoder_layer1(fusion1)

        fusion2 = sparsed_feature8 + decoder_feature1
        decoder_feature2 = self.decoder_layer2(fusion2)

        fusion3 = sparsed_feature6 + decoder_feature2
        decoder_feature3 = self.decoder_layer3(fusion3)

        fusion4 = sparsed_feature4 + decoder_feature3
        decoder_feature4 = self.decoder_layer4(fusion4)

        fusion5 = sparsed_feature2 + decoder_feature4
        decoder_feature5 = self.decoder_layer5(fusion5)

        depth_output = self.decoder_layer6(decoder_feature5)
        d_depth, d_conf = torch.chunk(depth_output, 2, dim=1)

        rgb_conf, semantic_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf, semantic_conf, d_conf), dim=1)), 3, dim=1)
        output = rgb_conf*rgb_depth + d_conf*d_depth + semantic_conf * semantic_depth
        #output =  d_conf*d_depth
        
        if(self.args.network_model == 'e'):
            return rgb_depth, semantic_depth, d_depth, output
        elif(self.args.dilation_rate == 1):
            return torch.cat((rgb_feature0_plus, decoder_feature5),1), output
        elif (self.args.dilation_rate == 2):
            return torch.cat((rgb_feature0_plus,semantic_feature_decoder5, decoder_feature5), 1), torch.cat((rgb_feature2_plus,semantic_feature_decoder4, decoder_feature4),1), \
             rgb_conf, semantic_conf, d_conf, rgb_depth, semantic_depth, d_depth,  output
        elif (self.args.dilation_rate == 4):
            return torch.cat((rgb_feature0_plus, decoder_feature5), 1), torch.cat((rgb_feature2_plus, decoder_feature4),1),\
                   torch.cat((rgb_feature4_plus, decoder_feature3), 1), output



# CSPN++ Implementation Borrowed from 

# https://github.com/JUGGHM/PENet_ICRA2021/blob/main/model.py




class A_CSPN_plus_plus(nn.Module):
    def __init__(self, args):
        super(A_CSPN_plus_plus, self).__init__()

        self.backbone = three_branch_bb(args)
        self.kernel_conf_layer = convbn(96, 3)
        self.mask_layer = convbn(96, 1)
        self.iter_guide_layer3 = CSPNGenerate(96, 3)
        self.iter_guide_layer5 = CSPNGenerate(96, 5)
        self.iter_guide_layer7 = CSPNGenerate(96, 7)

        self.kernel_conf_layer_s2 = convbn(192, 3)
        self.mask_layer_s2 = convbn(192, 1)
        self.iter_guide_layer3_s2 = CSPNGenerate(192, 3)
        self.iter_guide_layer5_s2 = CSPNGenerate(192, 5)
        self.iter_guide_layer7_s2 = CSPNGenerate(192, 7)

        self.dimhalf_s2 = convbnrelu(192, 96, 1, 1, 0)
        self.att_12 = convbnrelu(192, 2)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downsample = SparseDownSampleClose(stride=2)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPN(3)
        self.CSPN5 = CSPN(5)
        self.CSPN7 = CSPN(7)

        weights_init(self)

    def forward(self, input):
        d = input['d']
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        feature_s1, feature_s2, rgb_conf, semantic_conf, d_conf, rgb_depth, semantic_depth, d_depth, coarse_depth = self.backbone(input)
        #feature_s1, feature_s2, coarse_depth = self.backbone(input)
        depth = coarse_depth

        d_s2, valid_mask_s2 = self.downsample(d, valid_mask)
        mask_s2 = self.mask_layer_s2(feature_s2)
        mask_s2 = torch.sigmoid(mask_s2)
        mask_s2 = mask_s2*valid_mask_s2

        kernel_conf_s2 = self.kernel_conf_layer_s2(feature_s2)
        kernel_conf_s2 = self.softmax(kernel_conf_s2)
        kernel_conf3_s2 = kernel_conf_s2[:, 0:1, :, :]
        kernel_conf5_s2 = kernel_conf_s2[:, 1:2, :, :]
        kernel_conf7_s2 = kernel_conf_s2[:, 2:3, :, :]

        mask = self.mask_layer(feature_s1)
        mask = torch.sigmoid(mask)
        mask = mask*valid_mask

        kernel_conf = self.kernel_conf_layer(feature_s1)
        kernel_conf = self.softmax(kernel_conf)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        feature_12 = torch.cat((feature_s1, self.upsample(self.dimhalf_s2(feature_s2))), 1)
        att_map_12 = self.softmax(self.att_12(feature_12))

        guide3_s2 = self.iter_guide_layer3_s2(feature_s2)
        guide5_s2 = self.iter_guide_layer5_s2(feature_s2)
        guide7_s2 = self.iter_guide_layer7_s2(feature_s2)
        guide3 = self.iter_guide_layer3(feature_s1)
        guide5 = self.iter_guide_layer5(feature_s1)
        guide7 = self.iter_guide_layer7(feature_s1)

        depth_s2 = depth
        depth_s2_00 = depth_s2[:, :, 0::2, 0::2]
        depth_s2_01 = depth_s2[:, :, 0::2, 1::2]
        depth_s2_10 = depth_s2[:, :, 1::2, 0::2]
        depth_s2_11 = depth_s2[:, :, 1::2, 1::2]

        depth_s2_00_h0 = depth3_s2_00 = depth5_s2_00 = depth7_s2_00 = depth_s2_00
        depth_s2_01_h0 = depth3_s2_01 = depth5_s2_01 = depth7_s2_01 = depth_s2_01
        depth_s2_10_h0 = depth3_s2_10 = depth5_s2_10 = depth7_s2_10 = depth_s2_10
        depth_s2_11_h0 = depth3_s2_11 = depth5_s2_11 = depth7_s2_11 = depth_s2_11

        for i in range(6):
            depth3_s2_00 = self.CSPN3(guide3_s2, depth3_s2_00, depth_s2_00_h0)
            depth3_s2_00 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_00
            depth5_s2_00 = self.CSPN5(guide5_s2, depth5_s2_00, depth_s2_00_h0)
            depth5_s2_00 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_00
            depth7_s2_00 = self.CSPN7(guide7_s2, depth7_s2_00, depth_s2_00_h0)
            depth7_s2_00 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_00

            depth3_s2_01 = self.CSPN3(guide3_s2, depth3_s2_01, depth_s2_01_h0)
            depth3_s2_01 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_01
            depth5_s2_01 = self.CSPN5(guide5_s2, depth5_s2_01, depth_s2_01_h0)
            depth5_s2_01 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_01
            depth7_s2_01 = self.CSPN7(guide7_s2, depth7_s2_01, depth_s2_01_h0)
            depth7_s2_01 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_01

            depth3_s2_10 = self.CSPN3(guide3_s2, depth3_s2_10, depth_s2_10_h0)
            depth3_s2_10 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_10
            depth5_s2_10 = self.CSPN5(guide5_s2, depth5_s2_10, depth_s2_10_h0)
            depth5_s2_10 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_10
            depth7_s2_10 = self.CSPN7(guide7_s2, depth7_s2_10, depth_s2_10_h0)
            depth7_s2_10 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_10

            depth3_s2_11 = self.CSPN3(guide3_s2, depth3_s2_11, depth_s2_11_h0)
            depth3_s2_11 = mask_s2*d_s2 + (1-mask_s2)*depth3_s2_11
            depth5_s2_11 = self.CSPN5(guide5_s2, depth5_s2_11, depth_s2_11_h0)
            depth5_s2_11 = mask_s2*d_s2 + (1-mask_s2)*depth5_s2_11
            depth7_s2_11 = self.CSPN7(guide7_s2, depth7_s2_11, depth_s2_11_h0)
            depth7_s2_11 = mask_s2*d_s2 + (1-mask_s2)*depth7_s2_11

        depth_s2_00 = kernel_conf3_s2*depth3_s2_00 + kernel_conf5_s2*depth5_s2_00 + kernel_conf7_s2*depth7_s2_00
        depth_s2_01 = kernel_conf3_s2*depth3_s2_01 + kernel_conf5_s2*depth5_s2_01 + kernel_conf7_s2*depth7_s2_01
        depth_s2_10 = kernel_conf3_s2*depth3_s2_10 + kernel_conf5_s2*depth5_s2_10 + kernel_conf7_s2*depth7_s2_10
        depth_s2_11 = kernel_conf3_s2*depth3_s2_11 + kernel_conf5_s2*depth5_s2_11 + kernel_conf7_s2*depth7_s2_11

        depth_s2[:, :, 0::2, 0::2] = depth_s2_00
        depth_s2[:, :, 0::2, 1::2] = depth_s2_01
        depth_s2[:, :, 1::2, 0::2] = depth_s2_10
        depth_s2[:, :, 1::2, 1::2] = depth_s2_11

        #feature_12 = torch.cat((feature_s1, self.upsample(self.dimhalf_s2(feature_s2))), 1)
        #att_map_12 = self.softmax(self.att_12(feature_12))
        refined_depth_s2 = depth*att_map_12[:, 0:1, :, :] + depth_s2*att_map_12[:, 1:2, :, :]
        #refined_depth_s2 = depth

        depth3 = depth5 = depth7 = refined_depth_s2

        #prop
        for i in range(6):
            depth3 = self.CSPN3(guide3, depth3, depth)
            depth3 = mask*d + (1-mask)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth)
            depth5 = mask*d + (1-mask)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth)
            depth7 = mask*d + (1-mask)*depth7

        refined_depth = kernel_conf3*depth3 + kernel_conf5*depth5 + kernel_conf7*depth7
       
        return rgb_conf, semantic_conf, d_conf, rgb_depth, semantic_depth, d_depth,coarse_depth,refined_depth
