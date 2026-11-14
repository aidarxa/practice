#pragma once
/** @file @brief Build-time PCUDA PCH: ABI and device helpers only, never SQL/parser/storage I/O. */
#include <pcuda.hpp>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include "core/module_contract_v2.h"
#include "core/text.h"
#include "core/pcuda_index.h"
#include "core/tagged_numeric.h"
#include "crystal/device.h"
