# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file holds utility functions for working with nq_data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import tensorflow as tf

from fat.fat_bert_nq.shortest_path_classifier import run_nq


def get_sharded_filename(data_dir, mode, task, split, filetype):
  return os.path.join(
      data_dir, "%s/nq-%s-%02d%02d.%s" % (mode, mode, task, split, filetype))


def get_nq_filename(data_dir, mode, task, filetype):
  return os.path.join(data_dir,
                      "%s/nq-%s-%02d.%s" % (mode, mode, task, filetype))


def get_nq_examples(input_file):
    with tf.gfile.Open(input_file, "r") as fp:
        l = fp.readline()
        for line in fp:
            yield run_nq.create_example_from_line(line)
