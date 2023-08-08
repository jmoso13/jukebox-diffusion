from jbdiff.utils import wget

v1_models_map = {

    "dd": {'downloaded': False,
                         'sha': "07a121730560d3ca217cd665630412e40d2c3d05b7629c4fc796c51148cdb9ee", 
                         'uri_list': ["https://huggingface.co/jmoso13/jukebox-diffusion/resolve/main/v1/epoch%3D2125-step%3D218000.ckpt"],
                         },
    "0": {'downloaded': False,
                         'sha': "1d4611627d190106351a6ebfc82733baaba5af05144c1f1fa8333057ad8d53dc", 
                         'uri_list': ["https://huggingface.co/jmoso13/jukebox-diffusion/resolve/main/v1/epoch%3D543-step%3D705000.ckpt"],
                         },
    "1": {'downloaded': False,
                         'sha': "6f74d02f2cdaec53d2e310184a294b3dda617ef012b2ec83b0bc23595a0bf27f", 
                         'uri_list': ["https://huggingface.co/jmoso13/jukebox-diffusion/resolve/main/v1/epoch%3D1404-step%3D455000.ckpt"],
                         },
    "2": {'downloaded': False,
                         'sha': "424917a12814cbfab58aaf7dcdc7ff53663fa965da525eeb3e0b219572307cb1", 
                         'uri_list': ["https://huggingface.co/jmoso13/jukebox-diffusion/resolve/main/v1/epoch%3D4938-step%3D400000_vqvae_add.ckpt"],
                         },
}

def main():
  for model, values in v1_models_map.items():
    wget(values['uri_list'][0])

if __name__ == "__main__":
    main()