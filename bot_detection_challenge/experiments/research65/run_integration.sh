#!/bin/sh
# MB is excluded: it failed the mapping-quality gate (dR2 +0.011 over context, shuffle control +0.008 MAE)
cd "$(dirname "$0")/.."
python research65/integrate.py E65_IB IB > research65/int_E65_IB.log 2>&1
python research65/integrate.py E65_BI BI > research65/int_E65_BI.log 2>&1
python research65/integrate.py E65_MIB MIB > research65/int_E65_MIB.log 2>&1
python research65/integrate.py E65_all IB BI MIB > research65/int_E65_all.log 2>&1
