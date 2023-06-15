import os


#----------------------------------------------------------------------------


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _path_exists(p):
  if p is not None:
    if os.path.exists(p):
      return p
    else:
      raise argparse.ArgumentTypeError('Input path does not exist.')
  return p


#----------------------------------------------------------------------------


def main():
  parser = argparse.ArgumentParser(
    description = 'Sample from JBDiffusion', 
    epilog=_examples, 
    formatter_class=argparse.RawDescriptionHelpFormatter
    )
  parser.add_argument('--sample-rate', help='Sample rate of the data/network', required=True, type=int)
  parser.add_argument('--log-to-wandb', help='T/F whether to log to weights and biases', default=False, metavar='BOOL', type=_str_to_bool)
  parser.add_argument('--batch-size', help='Number of examples you want in batch', default=32, type=int)
  parser.add_argument('--lowest-level-pkl', help='Location of lowest level network pkl for use in sampling', default=None, metavar='FILE', type=_path_exists)
  parser.add_argument('--middle-level-pkl', help='Location of middle level network pkl for use in sampling', default=None, metavar='FILE', type=_path_exists)
  parser.add_argument('--highest-level-pkl', help='Location of highest level network pkl for use in sampling', default=None, metavar='FILE', type=_path_exists)
  args = parser.parse_args()


  run(**vars(args))


#----------------------------------------------------------------------------


if __name__ == "__main__":
    main()


#----------------------------------------------------------------------------
