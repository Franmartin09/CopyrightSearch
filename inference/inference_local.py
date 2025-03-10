import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime, timedelta

def extract_generated_response(generated_text):
    # Separar la respuesta del usuario y del assistant
    parts = generated_text.split("<|im_start|>assistant\n")
    
    if len(parts) > 1:
        response = parts[1].split("<|im_end|>")[0].strip()
    else:
        response = generated_text
    
    return response

# Configurar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ruta local del modelo finetuneado
local_model_path = "./CopyrightSearchLM"

# Cargar modelo y tokenizer desde el directorio local
model = AutoModelForCausalLM.from_pretrained(local_model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# ============================
# PRUEBA 1 EJEMPLOS QUE ESTAN EN EL TRAINING....
# ============================
array_ejemplos=[
    {
        "file": "kfd_kernel_queue.c",
        "comment": "/*\n * Copyright 2014 Advanced Micro Devices, Inc.\n *\n * Permission is hereby granted, free of charge, to any person obtaining a\n * copy of this software and associated documentation files (the \"Software\"),\n * to deal in the Software without restriction, including without limitation\n * the rights to use, copy, modify, merge, publish, distribute, sublicense,\n * and/or sell copies of the Software, and to permit persons to whom the\n * Software is furnished to do so, subject to the following conditions:\n *\n * The above copyright notice and this permission notice shall be included in\n * all copies or substantial portions of the Software.\n *\n * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL\n * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR\n * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,\n * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR\n * OTHER DEALINGS IN THE SOFTWARE.\n *\n */\n/* assign HIQ to HQD */\n/* allocate fence for DIQ */\n/* When rptr == wptr, the buffer is empty.\n\t * When rptr == wptr + 1, the buffer is full.\n\t * It is always rptr that advances to the position of wptr, rather than\n\t * the opposite. So we can only use up to queue_size_dwords - 1 dwords.\n\t */\n/*\n\t\t * make sure calling functions know\n\t\t * acquire_packet_buffer() failed\n\t\t */\n/* make sure after rolling back to position 0, there is\n\t\t * still enough space.\n\t\t */\n/* fill nops, roll back and start at position 0 */\n/* FIXME: Can this test be removed? */",
        "result": "Copyright (c) 2014 Advanced Micro Devices, Inc.,\n"
    },
    {
        "file": "osl.c",
        "comment": "// SPDX-License-Identifier: GPL-2.0-or-later\n/*\n *  acpi_osl.c - OS-dependent functions ($Revision: 83 $)\n *\n *  Copyright (C) 2000       Andrew Henroid\n *  Copyright (C) 2001, 2002 Andy Grover <andrew.grover@intel.com>\n *  Copyright (C) 2001, 2002 Paul Diefenbaugh <paul.s.diefenbaugh@intel.com>\n *  Copyright (c) 2008 Intel Corporation\n *   Author: Matthew Wilcox <willy@linux.intel.com>\n */\n/* stuff for debugger support */\n/*ENABLE_DEBUGGER */\n/*\n * This list of permanent mappings is for memory that may be accessed from\n * interrupt context, where we can't do the ioremap().\n */\n/* Handle possible alignment issues */\n/* Resources are never freed */\n/* Length of GPE blocks must be a non-negative multiple of 2 */\n/*\n\t * We may have been provided with an RSDP on the command line,\n\t * but if a malicious user has done so they may be pointing us\n\t * at modified ACPI tables that could alter kernel behaviour -\n\t * so, we check the lockdown status before making use of\n\t * it. If we trust it then also stash it in an architecture\n\t * specific location (if appropriate) so it can be carried\n\t * over further kexec()s.\n\t */\n/* Must be called with 'acpi_ioremap_lock' or RCU read lock held. */\n/* Must be called with 'acpi_ioremap_lock' or RCU read lock held. */\n/* Must be called with 'acpi_ioremap_lock' or RCU read lock held. */\n/* ioremap will take care of cache attributes */\n/**\n * acpi_os_map_iomem - Get a virtual address for a given physical address range.\n * @phys: Start of the physical address range to map.\n * @size: Size of the physical address range to map.\n *\n * Look up the given physical address range in the list of existing ACPI memory\n * mappings.  If found, get a reference to it and return a pointer to it (its\n * virtual address).  If not found, map it, add it to that list and return a\n * pointer to it.\n *\n * During early init (when acpi_permanent_mmap has not been set yet) this\n * routine simply calls __acpi_map_table() to get the job done.\n */\n/* Check if there's a suitable mapping already. */\n/* Must be called with mutex_lock(&acpi_ioremap_lock) */\n/**\n * acpi_os_unmap_iomem - Drop a memory mapping reference.\n * @virt: Start of the address range to drop a reference to.\n * @size: Size of the address range to drop a reference to.\n *\n * Look up the given virtual address range in the list of existing ACPI memory\n * mappings, drop a reference to it and unmap it if there are no more active\n * references to it.\n *\n * During early init (when acpi_permanent_mmap has not been set yet) this\n * routine simply calls __acpi_unmap_table() to get the job done.  Since\n * __acpi_unmap_table() is an __init function, the __ref annotation is needed\n * here.\n */\n/* Handle possible alignment issues */\n/* Handle possible alignment issues */\n/*\n\t * ACPI interrupts different from the SCI in our copy of the FADT are\n\t * not supported.\n\t */\n/*\n * Running in interpreter thread context, safe to sleep\n */\n/*\n * Support ACPI 3.0 AML Timer operand. Returns a 64-bit free-running,\n * monotonically increasing timer with 100ns granularity. Do not use\n * ktime_get() to implement this function because this function may get\n * called after timekeeping has been suspended. Note: calling this function\n * after timekeeping has been suspended may lead to unexpected results\n * because when timekeeping is suspended the jiffies counter is not\n * incremented. See also timekeeping_suspend().\n */\n/*******************************************************************************\n *\n * FUNCTION:    acpi_os_execute\n *\n * PARAMETERS:  Type               - Type of the callback\n *              Function           - Function to be executed\n *              Context            - Function parameters\n *\n * RETURN:      Status\n *\n * DESCRIPTION: Depending on type, either queues function for deferred execution or\n *              immediately executes function on a separate thread.\n *\n ******************************************************************************/\n/*\n\t * Allocate/initialize DPC structure.  Note that this memory will be\n\t * freed by the callee.  The kernel handles the work_struct list  in a\n\t * way that allows us to also free its memory inside the callee.\n\t * Because we may want to schedule several tasks with different\n\t * parameters we can't use the approach some kernel code uses of\n\t * having a static work_struct.\n\t */\n/*\n\t * To prevent lockdep from complaining unnecessarily, make sure that\n\t * there is a different static lockdep key for each workqueue by using\n\t * INIT_WORK() for each of them separately.\n\t */\n/*\n\t * On some machines, a software-initiated SMI causes corruption unless\n\t * the SMI runs on CPU 0.  An SMI can be initiated by any AML, but\n\t * typically it's done in GPE-related methods that are run via\n\t * workqueues, so we can avoid the known corruption cases by always\n\t * queueing on CPU 0.\n\t */\n/*\n\t * Make sure the GPE handler or the fixed event handler is not used\n\t * on another CPU after removal.\n\t */\n/*\n\t * We can't run hotplug code in kacpid_wq/kacpid_notify_wq etc., because\n\t * the hotplug code may call driver .remove() functions, which may\n\t * invoke flush_scheduled_work()/acpi_os_wait_events_complete() to flush\n\t * these workqueues.\n\t */\n/*\n * TODO: A better way to delete semaphores?  Linux doesn't have a\n * 'delete_semaphore()' function -- may result in an invalid\n * pointer dereference for non-synchronized consumers.\tShould\n * we at least check for blocked threads and signal/cancel them?\n */\n/*\n * TODO: Support for units > 1?\n */\n/*\n * TODO: Support for units > 1?\n */\n/* remove the CR kdb includes */\n/*\n\t\t * AML Breakpoint\n\t\t * ACPI spec. says to treat it as a NOP unless\n\t\t * you are debugging.  So if/when we integrate\n\t\t * AML debugger into the kernel debugger its\n\t\t * hook will go here.  But until then it is\n\t\t * not useful to print anything on breakpoints.\n\t\t */\n/*\n * Disable the auto-serialization of named objects creation methods.\n *\n * This feature is enabled by default.  It marks the AML control methods\n * that contain the opcodes to create named objects as \"Serialized\".\n */\n/* Check of resource interference between native drivers and ACPI\n * OperationRegions (SystemIO and System Memory only).\n * IO ports and memory declared in ACPI might be used by the ACPI subsystem\n * in arbitrary AML code and can interfere with legacy drivers.\n * acpi_enforce_resources= can be set to:\n *\n *   - strict (default) (2)\n *     -> further driver trying to access the resources will not load\n *   - lax              (1)\n *     -> further driver trying to access the resources will load, but you\n *     get a system message that something might go wrong...\n *\n *   - no               (0)\n *     -> ACPI Operation Region resources will not be registered\n *\n */\n/* Check for resource conflicts between ACPI OperationRegions and native\n * drivers */\n/**\n * acpi_release_memory - Release any mappings done to a memory region\n * @handle: Handle to namespace node\n * @res: Memory resource\n * @level: A level that terminates the search\n *\n * Walks through @handle and unmaps all SystemMemory Operation Regions that\n * overlap with @res and that have already been activated (mapped).\n *\n * This is a helper that allows drivers to place special requirements on memory\n * region that may overlap with operation regions, primarily allowing them to\n * safely map the region as non-cached memory.\n *\n * The unmapped Operation Regions will be automatically remapped next time they\n * are called, so the drivers do not need to do anything else.\n */\n/*\n * Let drivers know whether the resource checks are effective\n */\n/*\n * Deallocate the memory for a spinlock.\n */\n/*\n * Acquire a spinlock.\n *\n * handle is a pointer to the spinlock_t.\n */\n/*\n * Release a spinlock. See above.\n */\n/*******************************************************************************\n *\n * FUNCTION:    acpi_os_create_cache\n *\n * PARAMETERS:  name      - Ascii name for the cache\n *              size      - Size of each cached object\n *              depth     - Maximum depth of the cache (in objects) <ignored>\n *              cache     - Where the new cache object is returned\n *\n * RETURN:      status\n *\n * DESCRIPTION: Create a cache object\n *\n ******************************************************************************/\n/*******************************************************************************\n *\n * FUNCTION:    acpi_os_purge_cache\n *\n * PARAMETERS:  Cache           - Handle to cache object\n *\n * RETURN:      Status\n *\n * DESCRIPTION: Free all objects within the requested cache.\n *\n ******************************************************************************/\n/*******************************************************************************\n *\n * FUNCTION:    acpi_os_delete_cache\n *\n * PARAMETERS:  Cache           - Handle to cache object\n *\n * RETURN:      Status\n *\n * DESCRIPTION: Free all objects within the requested cache and delete the\n *              cache object.\n *\n ******************************************************************************/\n/*******************************************************************************\n *\n * FUNCTION:    acpi_os_release_object\n *\n * PARAMETERS:  Cache       - Handle to cache object\n *              Object      - The object to be released\n *\n * RETURN:      None\n *\n * DESCRIPTION: Release an object to the specified cache.  If cache is full,\n *              the object is deleted.\n *\n ******************************************************************************/\n/*\n\t\t * Use acpi_os_map_generic_address to pre-map the reset\n\t\t * register if it's in system memory.\n\t\t */",
        "result": "Copyright (C) 2000 Andrew Henroid\nCopyright (C) 2001, 2002 Andy Grover <andrew.grover@intel.com>\nCopyright (C) 2001, 2002 Paul Diefenbaugh <paul.s.diefenbaugh@intel.com>\nCopyright (c) 2008 Intel Corporation\nAuthor: Matthew Wilcox <willy@linux.intel.com>\n"
    },
    {
        "file": "isoch.c",
        "comment": "// SPDX-License-Identifier: GPL-2.0\n/*\n * Setup routines for AGP 3.5 compliant bridges.\n */\n/* Generic AGP 3.5 enabling routines */\n/*\n * Initialize all isochronous transfer parameters for an AGP 3.0\n * node (i.e. a host bridge in combination with the adapters\n * lying behind it...)\n */\n/*\n\t * Convenience structure to make the calculations clearer\n\t * here.  The field names come straight from the AGP 3.0 spec.\n\t */\n/*\n\t * We'll work with an array of isoch_data's (one for each\n\t * device in dev_list) throughout this function.\n\t */\n/*\n\t * Sort the device list by maxbw.  We need to do this because the\n\t * spec suggests that the devices with the smallest requirements\n\t * have their resources allocated first, with all remaining resources\n\t * falling to the device with the largest requirement.\n\t *\n\t * We don't exactly do this, we divide target resources by ndevs\n\t * and split them amongst the AGP 3.0 devices.  The remainder of such\n\t * division operations are dropped on the last device, sort of like\n\t * the spec mentions it should be done.\n\t *\n\t * We can't do this sort when we initially construct the dev_list\n\t * because we don't know until this function whether isochronous\n\t * transfers are enabled and consequently whether maxbw will mean\n\t * anything.\n\t */\n/* Extract power-on defaults from the target */\n/*\n\t * Extract power-on defaults for each device in dev_list.  Along\n\t * the way, calculate the total isochronous bandwidth required\n\t * by these devices and the largest requested payload size.\n\t */\n/* Check if this configuration has any chance of working */\n/*\n\t * Write the calculated payload size into the target's NICMD\n\t * register.  Doing this directly effects the ISOCH_N value\n\t * in the target's NISTAT register, so we need to do this now\n\t * to get an accurate value for ISOCH_N later.\n\t */\n/* Reread the target's ISOCH_N */\n/* Calculate the minimum ISOCH_N needed by each master */\n/* Exit if the minimal ISOCH_N allocation among the masters is more\n\t * than the target can handle. */\n/* Calculate left over ISOCH_N capability in the target.  We'll give\n\t * this to the hungriest device (as per the spec) */\n/*\n\t * Calculate the minimum isochronous RQ depth needed by each master.\n\t * Along the way, distribute the extra ISOCH_N capability calculated\n\t * above.\n\t */\n/*\n\t\t * This is a little subtle.  If ISOCH_Y > 64B, then ISOCH_Y\n\t\t * byte isochronous writes will be broken into 64B pieces.\n\t\t * This means we need to budget more RQ depth to account for\n\t\t * these kind of writes (each isochronous write is actually\n\t\t * many writes on the AGP bus).\n\t\t */\n/* Figure the number of isochronous and asynchronous RQ slots the\n\t * target is providing. */\n/* Exit if the minimal RQ needs of the masters exceeds what the target\n\t * can provide. */\n/* Calculate asynchronous RQ capability in the target (per master) as\n\t * well as the total number of leftover isochronous RQ slots. */\n/* Distribute the extra RQ slots calculated above and write our\n\t * isochronous settings out to the actual devices. */\n/*\n * This function basically allocates request queue slots among the\n * AGP 3.0 systems in nonisochronous nodes.  The algorithm is\n * pretty stupid, divide the total number of RQ slots provided by the\n * target by ndevs.  Distribute this many slots to each AGP 3.0 device,\n * giving any left over slots to the last device in dev_list.\n */\n/*\n * Fully configure and enable an AGP 3.0 host bridge and all the devices\n * lying behind it.\n */\n/* Extract some power-on defaults from the target */\n/* isoch xfers not available, bail out. */\n/*\n\t * Allocate a head for our AGP 3.5 device list\n\t * (multiple AGP v3 devices are allowed behind a single bridge).\n\t */\n/* Find all AGP devices, and add them to dev_list. */\n/* Bridge */\n/* Skip bridges. We should call this function for each one. */\n/* Unclassified device */\n/* Don't know what this is, but log it for investigation. */\n/* Display controller */\n/* Multimedia controller */\n/*\n\t * Take an initial pass through the devices lying behind our host\n\t * bridge.  Make sure each one is actually an AGP 3.0 device, otherwise\n\t * exit with an error message.  Along the way store the AGP 3.0\n\t * cap_ptr for each device\n\t */\n/*\n\t * Call functions to divide target resources amongst the AGP 3.0\n\t * masters.  This process is dramatically different depending on\n\t * whether isochronous transfers are supported.\n\t */\n/* Be sure to free the dev_list */",
        "result": "None\n"
    },
    {
        "file": "nfcon.c",
        "comment": "/*\n * ARAnyM console driver\n *\n * This file is subject to the terms and conditions of the GNU General Public\n * License.  See the file COPYING in the main directory of this archive\n * for more details.\n */\n/* !MODULE */",
        "result": "None\n"
    },
]

print("Ejemplos del dataset training: ")
tiempo_suma=timedelta()
for idx, ejemplo in enumerate(array_ejemplos):
    instruction = f"Extract copyright from this text:\n{ejemplo['comment']}\n"
    messages = [{"role": "user", "content": instruction}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenizar la entrada y moverla al dispositivo
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    start = datetime.now()
    # Generar la respuesta
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    generated_text = tokenizer.decode(outputs[0])
    end = datetime.now()
    assistant_response = extract_generated_response(generated_text)

    print(f"Ejemplo {idx} - Predicción:")
    print(assistant_response + '\n')
    print(f"Tiempo de ejecucion : {end-start}")
    tiempo_suma=tiempo_suma+(end-start)
# ============================
# PRUEBA 2 EJEMPLOS QUE NO ESTAN EN EL TRAINING....
# ============================

array_ejemplos=[
    {
        "file": "vmcs_shadow_fields.h",
        "comment": "/*\n * We do NOT shadow fields that are modified when L0\n * traps and emulates any vmx instruction (e.g. VMPTRLD,\n * VMXON...) executed by L1.\n * For example, VM_INSTRUCTION_ERROR is read\n * by L1 if a vmx instruction fails (part of the error path).\n * Note the code assumes this logic. If for some reason\n * we start shadowing these fields then we need to\n * force a shadow sync when L0 emulates vmx instructions\n * (e.g. force a sync if VM_INSTRUCTION_ERROR is modified\n * by nested_vmx_failValid)\n *\n * When adding or removing fields here, note that shadowed\n * fields must always be synced by prepare_vmcs02, not just\n * prepare_vmcs02_full.\n */\n/*\n * Keeping the fields ordered by size is an attempt at improving\n * branch prediction in vmcs_read_any and vmcs_write_any.\n */\n/* 16-bits */\n/* 32-bits */\n/* Natural width */\n/* 64-bit */",
        "result": "None\n"
    },
    {
        "file": "floppy.h",
        "comment": "/* SPDX-License-Identifier: GPL-2.0-or-later */\n/*    Architecture specific parts of the Floppy driver\n *\n *    Linux/PA-RISC Project (http://www.parisc-linux.org/)\n *    Copyright (C) 2000 Matthew Wilcox (willy a debian . org)\n *    Copyright (C) 2000 Dave Kennedy\n */\n/*\n * The DMA channel used by the floppy controller cannot access data at\n * addresses >= 16MB\n *\n * Went back to the 1MB limit, as some people had problems with the floppy\n * driver otherwise. It doesn't matter much for performance anyway, as most\n * floppy accesses go through the track buffer.\n */\n/* actual, physical DMA */\n/* Lies.  Floppy controller is memory mapped, not io mapped */\n/* __ASM_PARISC_FLOPPY_H */",
        "result": "Copyright (C) 2000 Matthew Wilcox (willy a debian . org), Copyright (C) 2000 Dave Kennedy\n"
    },
    {
        "file": "mpc85xx_edac.h",
        "comment": "/*\n * Freescale MPC85xx Memory Controller kernel module\n * Author: Dave Jiang <djiang@mvista.com>\n *\n * 2006-2007 (c) MontaVista Software, Inc. This file is licensed under\n * the terms of the GNU General Public License version 2. This program\n * is licensed \"as is\" without any warranty of any kind, whether express\n * or implied.\n *\n */\n/*\n * L2 Err defines\n */\n/* Error Interrupt Enable */\n/* Error Detect */\n/*\n * PCI Err defines\n */",
        "result": "Author: Dave Jiang djiang@mvista.com,\nCopyright (c) 2006-2007 MontaVista Software, Inc.,\n"
    },
    {
        "file": "pdc.h",
        "comment": "/* SPDX-License-Identifier: GPL-2.0 */\n/* cell number the CPU runs on (PAT) */\n/* cell location of CPU (PAT)\t   */\n/* PDC capabilities (PAT) */\n/* Values for pdc_type */\n/* 64-bit PAT-PDC */\n/* 32-bit, but supports PDC_SYSTEM_MAP */\n/* Doesn't support SYSTEM_MAP */\n/* in pdc_console.c */\n/* in inventory.c */\n/* wrapper-functions from pdc.c */\n/* !CONFIG_PA20 */\n/* in pdt.c */\n/* !defined(__ASSEMBLY__) */\n/* _PARISC_PDC_H */",
        "result": "None\n"
    },
]

print("Ejemplos que no estan en el training!")
start_nonexist = datetime.now()
for idx, ejemplo in enumerate(array_ejemplos):
    instruction = f"Extract copyright from this text:\n{ejemplo['comment']}\n"
    messages = [{"role": "user", "content": instruction}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenizar la entrada y moverla al dispositivo
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    start = datetime.now()
    # Generar la respuesta
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    generated_text = tokenizer.decode(outputs[0])
    end = datetime.now()
    assistant_response = extract_generated_response(generated_text)

    print(f"Ejemplo {idx} - Respuesta Generada:")
    print(assistant_response + '\n')
    print(f"Tiempo de ejecucion : {end-start}")
    tiempo_suma=tiempo_suma+(end-start)

print(f"Tiempo total de las 8 ejecuciones: {tiempo_suma}")







