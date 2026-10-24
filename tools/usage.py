"""Read-only resource snapshot for the lab's usage panel: host, model process and GPU."""
import os
import platform
import re
import shutil
import socket
import subprocess

UNITS = {'B': 1, 'K': 1024, 'M': 1024 ** 2, 'G': 1024 ** 3, 'T': 1024 ** 4}


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ''


def parse_size(text):
    """Parse top's memory column, such as '16G', '512M+' or '1504K'."""
    match = re.fullmatch(r'([\d.]+)([BKMGT])[+-]?', text.strip())
    return int(float(match.group(1)) * UNITS[match.group(2)]) if match else None


def parse_nvidia(gpus_csv, apps_csv, visible, pid):
    """GPUs this process may use (CUDA_VISIBLE_DEVICES by UUID or index) and its memory on them."""
    wanted = [token.strip() for token in visible.split(',') if token.strip()] if visible else None
    used_by_pid = sum(int(memory) for app_pid, memory in
                      (line.split(', ') for line in apps_csv.splitlines() if line.strip())
                      if pid is not None and app_pid == str(pid))
    gpus = []
    for line in gpus_csv.splitlines():
        if not line.strip():
            continue
        index, uuid, name, utilization, used, total = line.split(', ')
        if wanted is not None and not any(uuid.startswith(token) or index == token for token in wanted):
            continue
        gpus.append({'name': name, 'kind': 'cuda', 'utilization_percent': float(utilization),
                     'memory_used_bytes': int(used) * 1024 ** 2, 'memory_total_bytes': int(total) * 1024 ** 2})
    if gpus:
        gpus[0]['process_memory_bytes'] = used_by_pid * 1024 ** 2
    return gpus


def parse_ioreg(text, chip, memory_total):
    """Apple GPU utilization and the unified memory its allocations hold."""
    utilization = re.search(r'"Device Utilization %"=(\d+)', text)
    # "Alloc" includes buffers such as loaded model weights; "In use" is only the active slice.
    allocated = re.search(r'"Alloc system memory"=(\d+)', text)
    if not utilization:
        return []
    return [{'name': chip + ' GPU', 'kind': 'mps', 'utilization_percent': float(utilization.group(1)),
             'memory_used_bytes': int(allocated.group(1)) if allocated else None,
             'memory_total_bytes': memory_total, 'unified_memory': True}]


def process_usage(pid):
    fields = run(['ps', '-o', '%cpu=,rss=', '-p', str(pid)]).split()
    if len(fields) != 2:
        return None
    usage = {'pid': pid, 'cpu_percent': float(fields[0]), 'memory_bytes': int(fields[1]) * 1024}
    if platform.system() == 'Darwin':
        # RSS omits Metal allocations; top's MEM is the full footprint, GPU memory included.
        rows = run(['top', '-l', '1', '-pid', str(pid), '-stats', 'mem']).strip().splitlines()
        footprint = parse_size(rows[-1]) if rows else None
        if footprint:
            usage['memory_bytes'] = footprint
    return usage


def gpus(pid):
    if shutil.which('nvidia-smi'):
        query = ['nvidia-smi', '--format=csv,noheader,nounits']
        return parse_nvidia(run(query + ['--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total']),
                            run(query + ['--query-compute-apps=pid,used_memory']),
                            os.environ.get('CUDA_VISIBLE_DEVICES'), pid)
    if platform.system() == 'Darwin':
        chip = run(['sysctl', '-n', 'machdep.cpu.brand_string']).strip() or 'Apple'
        total = run(['sysctl', '-n', 'hw.memsize']).strip()
        return parse_ioreg(run(['ioreg', '-r', '-d', '1', '-w', '0', '-c', 'IOAccelerator']),
                           chip, int(total) if total.isdigit() else None)
    return []


def sample(pid=None):
    return {'host': socket.gethostname().split('.')[0], 'system': platform.system(),
            'process': process_usage(pid) if pid else None, 'gpus': gpus(pid)}
