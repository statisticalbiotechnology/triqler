#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Convenience wrapper for running triqler directly from source tree."""

import sys
from triqler.triqler import main

if __name__ == '__main__':
    main(sys.argv[1:])
