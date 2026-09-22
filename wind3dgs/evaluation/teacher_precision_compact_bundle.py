"""완료 결과 ZIP의 동일 바이트 중복 제거. 원본 manifest로 무손실 복원 검증."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

RESTORE=r'''"""표준 Python만 필요. 원래 완료 결과의 모든 파일을 복원하고 SHA-256을 검증한다."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path,PurePosixPath

def safe(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('unsafe path: '+name)
    return name

def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(a.archive) as z:
        for item in z.infolist():safe(item.filename)
        z.extractall(a.output)
    mapping=json.loads((a.output/'duplicate_files.json').read_text())
    for target,source in mapping.items():
        src=a.output/safe(source);dst=a.output/safe(target)
        if dst.exists():raise FileExistsError(dst)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    manifest=json.loads((a.output/'experiment_manifest.json').read_text())['files']
    for name,expected in manifest.items():
        with (a.output/safe(name)).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:raise ValueError('SHA-256 mismatch: '+name)
    print('원본 파일 전체 복원·SHA-256 검증 완료:',len(manifest))
if __name__=='__main__':main()
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    manifest=json.loads((a.source/'experiment_manifest.json').read_text())['files']
    canonical={};stored={};duplicates={}
    for name,h in manifest.items():
        if h in canonical:duplicates[name]=canonical[h]
        else:canonical[h]=name;stored[name]=h
    README='''# 업로드용 무손실 압축본

원래 ZIP의 동일 내용 파일은 한 번만 보관한다. 원시 수치 데이터/정밀도는 바꾸지 않았다.
큰 JSON에는 ZIP LZMA를 사용하므로 표준 Python zipfile로 열 수 있다.
report.md는 압축본에서도 바로 읽을 수 있다. 분석 전 아래 복원 명령을 권장한다.

같이 제공된 restore_completed_v3.py를 ZIP과 같은 폴더에 놓고 Python 3.11 이상에서 실행한다.

    python restore_completed_v3.py completed_dual_gpu_20260916_compact.zip restored_v3

스크립트가 원래 폴더 구조/22,122개 manifest 파일을 복원하고 모든 SHA-256을 검증한다.
ZIP만 업로드했으면 표준 Python zipfile로 내부 restore_completed_v3.py를 먼저 꺼내 실행한다.
출력 폴더는 새 폴더여야 한다. 중복 관계는 duplicate_files.json에 있다.
원본 큰 ZIP과 source 결과는 별도로 보존돼 있다.
'''
    with zipfile.ZipFile(a.out,'w') as z:
        for i,name in enumerate(stored):
            path=a.source/name
            compression=zipfile.ZIP_LZMA if path.suffix=='.json' and path.stat().st_size>=1000000 else zipfile.ZIP_DEFLATED
            z.write(path,name,compress_type=compression,compresslevel=9 if compression==zipfile.ZIP_DEFLATED else None)
            if i%400==0:print('압축',i,'/',len(stored),flush=True)
        z.write(a.source/'experiment_manifest.json','experiment_manifest.json',compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
        z.writestr('duplicate_files.json',json.dumps(duplicates,ensure_ascii=False,indent=2),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
        z.writestr('restore_completed_v3.py',RESTORE,compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('COMPACT_README.md',README,compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('compact_manifest.json',json.dumps(dict(lossless=True,stored_files=len(stored),duplicate_files=len(duplicates),original_files=len(manifest),
            compression='ZIP DEFLATE plus LZMA for large JSON',original_manifest_sha256=hashlib.sha256((a.source/'experiment_manifest.json').read_bytes()).hexdigest()),indent=2),compress_type=zipfile.ZIP_DEFLATED)
    print('원본 해시 대조 검증',flush=True)
    with zipfile.ZipFile(a.out) as z:
        assert z.testzip() is None
        for name,expected in stored.items():
            with z.open(name) as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=expected:raise ValueError(name)
        for name,ref in duplicates.items():assert manifest[name]==stored[ref]
    a.out.with_name('restore_completed_v3.py').write_text(RESTORE)
    with a.out.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    a.out.with_suffix('.zip.sha256').write_text(h+'  '+a.out.name+'\n')
    size=a.out.stat().st_size
    a.out.with_suffix('.verification.json').write_text(json.dumps(dict(bytes=size,below_512_decimal_MB=size<512000000,crc='passed',all_original_hashes_represented='passed',stored=len(stored),duplicates=len(duplicates)),indent=2)+'\n')
    print('완료',size,'bytes; 512MB 미만:',size<512000000,flush=True)
if __name__=='__main__':main()
