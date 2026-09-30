#!/usr/bin/env bash

# Download WR2 experiment artifacts from a remote checkout.
#
# Common forms:
#   ./scripts/scp_from_remote.sh --latest
#   ./scripts/scp_from_remote.sh --latest 3
#   ./scripts/scp_from_remote.sh --latest servo_sysid 3
#   ./scripts/scp_from_remote.sh --latest training 2
#   ./scripts/scp_from_remote.sh --latest walking_agent
#   ./scripts/scp_from_remote.sh --list servo_sysid
#   ./scripts/scp_from_remote.sh --copy servo_sysid htd45h-unit-a
#   ./scripts/scp_from_remote.sh results/servo_sysid/deployment_spec.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE_USER="${WR2_REMOTE_USER:-leeygang}"
REMOTE_HOST="${WR2_REMOTE_HOST:-linux-pc.local}"
REMOTE_BASE="${WR2_REMOTE_BASE:-}"
REMOTE_PORT="${WR2_REMOTE_PORT:-}"
SSH_CONNECT_TIMEOUT="${WR2_SSH_CONNECT_TIMEOUT:-10}"
SSH_CONTROL_DIR="${WR2_SSH_CONTROL_DIR:-/tmp/wr2ssh-$UID}"
DRY_RUN=false
POSITIONAL=()
POSITIONAL_COUNT=0

usage() {
    cat <<'EOF'
Usage:
  ./scripts/scp_from_remote.sh --latest [N]
  ./scripts/scp_from_remote.sh --latest <group> [N]
  ./scripts/scp_from_remote.sh --list [group [N]]
  ./scripts/scp_from_remote.sh --groups
  ./scripts/scp_from_remote.sh --copy <group> <name>
  ./scripts/scp_from_remote.sh <repository-relative-path>

Result groups live below results/ on both machines. Aliases:
  training -> wr2_walking
  walking_agent -> wr2_walking_agent
  sysid    -> servo_sysid

Connection options may appear anywhere:
  --linux-pc            Use LINUX_PUBLIC_IP and optional LINUX_PUBLIC_PORT
  --wrdev               Use leeygang@wrdev.local (deployment host)
  --host <host>          Remote host (default: linux-pc.local)
  --user <user>          Remote user (default: leeygang)
  --port <port>          SSH port
  --remote-base <path>   Remote WR2 checkout
  --public [IPv4]        Use WR2_PUBLIC_IP or LINUX_PUBLIC_IP
  --dry-run              Resolve and print transfers without copying

Environment overrides:
  WR2_REMOTE_HOST, WR2_REMOTE_USER, WR2_REMOTE_PORT, WR2_REMOTE_BASE,
  WR2_WRDEV_BASE, WR2_PUBLIC_IP, WR2_PUBLIC_PORT,
  WR2_SSH_CONNECT_TIMEOUT, WR2_SSH_CONTROL_DIR

Examples:
  ./scripts/scp_from_remote.sh --latest servo_sysid
  ./scripts/scp_from_remote.sh --latest servo_sysid 3
  ./scripts/scp_from_remote.sh --latest training 2
  ./scripts/scp_from_remote.sh --latest walking_agent
  ./scripts/scp_from_remote.sh --copy servo_sysid htd45h-unit-a
  ./scripts/scp_from_remote.sh --host gpu-box --latest contact_tests 4
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

select_linux_pc() {
    if [ -n "${LINUX_PUBLIC_IP:-}" ]; then
        REMOTE_HOST="$LINUX_PUBLIC_IP"
    elif [ -n "${WR2_PUBLIC_IP:-}" ]; then
        REMOTE_HOST="$WR2_PUBLIC_IP"
    else
        fail "--linux-pc requires LINUX_PUBLIC_IP (or WR2_PUBLIC_IP)"
    fi
    if [ -z "$REMOTE_PORT" ]; then
        REMOTE_PORT="${LINUX_PUBLIC_PORT:-${WR2_PUBLIC_PORT:-}}"
    fi
}

is_positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -ge 1 ]
}

validate_component() {
    local value="$1"
    local label="$2"
    if [ -z "$value" ] || ! [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
        fail "invalid $label '$value'"
    fi
}

validate_relative_path() {
    local value="$1"
    if [ -z "$value" ] || [[ "$value" = /* ]] || [[ "$value" == *$'\n'* ]] || \
       [[ "/$value/" == *"/../"* ]] || [[ "/$value/" == *"/./"* ]] || \
       ! [[ "$value" =~ ^[A-Za-z0-9._+/-]+$ ]]; then
        fail "invalid repository-relative path '$value'"
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --linux-pc)
            select_linux_pc
            shift
            ;;
        --wrdev)
            REMOTE_USER="leeygang"
            REMOTE_HOST="wrdev.local"
            if [ -z "$REMOTE_BASE" ]; then
                REMOTE_BASE="${WR2_WRDEV_BASE:-/home/leeygang/projects/WildRobot2}"
            fi
            shift
            ;;
        --host)
            [ "$#" -ge 2 ] || fail "--host requires a value"
            REMOTE_HOST="$2"
            shift 2
            ;;
        --user)
            [ "$#" -ge 2 ] || fail "--user requires a value"
            REMOTE_USER="$2"
            shift 2
            ;;
        --port)
            [ "$#" -ge 2 ] || fail "--port requires a value"
            REMOTE_PORT="$2"
            shift 2
            ;;
        --remote-base)
            [ "$#" -ge 2 ] || fail "--remote-base requires a value"
            REMOTE_BASE="$2"
            shift 2
            ;;
        --public)
            shift
            if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
                REMOTE_HOST="$1"
                shift
                if [ -z "$REMOTE_PORT" ]; then
                    REMOTE_PORT="${LINUX_PUBLIC_PORT:-${WR2_PUBLIC_PORT:-}}"
                fi
            else
                select_linux_pc
            fi
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            POSITIONAL_COUNT=$((POSITIONAL_COUNT + 1))
            shift
            ;;
    esac
done

if [ "$POSITIONAL_COUNT" -eq 0 ]; then
    usage
    exit 1
fi
set -- "${POSITIONAL[@]}"

if [ -z "$REMOTE_BASE" ]; then
    REMOTE_BASE="/home/$REMOTE_USER/projects/WildRobot2"
fi
if ! [[ "$REMOTE_USER" =~ ^[A-Za-z0-9._-]+$ ]]; then
    fail "invalid remote user '$REMOTE_USER'"
fi
if ! [[ "$REMOTE_HOST" =~ ^[A-Za-z0-9._:-]+$ ]]; then
    fail "invalid remote host '$REMOTE_HOST'"
fi
if [ -n "$REMOTE_PORT" ] && ! is_positive_integer "$REMOTE_PORT"; then
    fail "SSH port must be a positive integer"
fi
if ! is_positive_integer "$SSH_CONNECT_TIMEOUT"; then
    fail "WR2_SSH_CONNECT_TIMEOUT must be a positive integer"
fi
if ! [[ "$REMOTE_BASE" =~ ^/[A-Za-z0-9._+/-]+$ ]] || [[ "/$REMOTE_BASE/" == *"/../"* ]]; then
    fail "invalid remote base '$REMOTE_BASE'"
fi
if ! [[ "$SSH_CONTROL_DIR" =~ ^/[A-Za-z0-9._+/-]+$ ]]; then
    fail "invalid SSH control directory '$SSH_CONTROL_DIR'"
fi

mkdir -p "$SSH_CONTROL_DIR"
chmod 700 "$SSH_CONTROL_DIR"
SSH_COMMON_ARGS=(
    -o "ConnectTimeout=$SSH_CONNECT_TIMEOUT"
    -o ControlMaster=auto
    -o ControlPersist=60
    -o "ControlPath=$SSH_CONTROL_DIR/%C"
)

REMOTE_TARGET="$REMOTE_USER@$REMOTE_HOST"

remote_ssh() {
    if [ -n "$REMOTE_PORT" ]; then
        ssh "${SSH_COMMON_ARGS[@]}" -p "$REMOTE_PORT" "$REMOTE_TARGET" "$@"
    else
        ssh "${SSH_COMMON_ARGS[@]}" "$REMOTE_TARGET" "$@"
    fi
}

remote_scp() {
    if [ -n "$REMOTE_PORT" ]; then
        scp "${SSH_COMMON_ARGS[@]}" -P "$REMOTE_PORT" "$@"
    else
        scp "${SSH_COMMON_ARGS[@]}" "$@"
    fi
}

remote_rsync() {
    local source="$1"
    local destination="$2"
    local ssh_transport="ssh -o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o ControlMaster=auto -o ControlPersist=60 -o ControlPath=$SSH_CONTROL_DIR/%C"
    if [ -n "$REMOTE_PORT" ]; then
        ssh_transport="$ssh_transport -p $REMOTE_PORT"
    fi
    rsync -az --partial --progress -e "$ssh_transport" "$source" "$destination"
}

resolve_group() {
    case "$1" in
        training) echo "wr2_walking" ;;
        walking_agent) echo "wr2_walking_agent" ;;
        sysid) echo "servo_sysid" ;;
        *) echo "$1" ;;
    esac
}

remote_rows() {
    local group="$1"
    local remote_dir="$REMOTE_BASE/results/$group"
    remote_ssh \
        "if [ ! -d '$remote_dir' ]; then echo 'Missing remote result group: $remote_dir' >&2; exit 4; fi; find '$remote_dir' -mindepth 1 -maxdepth 1 -printf '%T@\t%y\t%f\n' | sort -nr"
}

copy_relative_path() {
    local relative_path="$1"
    local remote_path="$REMOTE_BASE/$relative_path"
    local local_path="$REPO_ROOT/$relative_path"
    validate_relative_path "$relative_path"

    if remote_ssh "[ -d '$remote_path' ]" 2>/dev/null; then
        echo "Directory: $REMOTE_TARGET:$remote_path -> $local_path"
        if [ "$DRY_RUN" = true ]; then
            return 0
        fi
        if command -v rsync >/dev/null 2>&1; then
            mkdir -p "$local_path"
            remote_rsync "$REMOTE_TARGET:$remote_path/" "$local_path/"
        else
            mkdir -p "$local_path"
            remote_scp -r "$REMOTE_TARGET:$remote_path/." "$local_path/"
        fi
    elif remote_ssh "[ -f '$remote_path' ]" 2>/dev/null; then
        echo "File:      $REMOTE_TARGET:$remote_path -> $local_path"
        if [ "$DRY_RUN" = true ]; then
            return 0
        fi
        mkdir -p "$(dirname "$local_path")"
        remote_scp "$REMOTE_TARGET:$remote_path" "$local_path"
    else
        echo "Missing remote path: $remote_path" >&2
        return 1
    fi
}

copy_bundle() {
    local group="$1"
    local name="$2"
    local copied=0
    local extension
    validate_component "$name" "result name"
    for extension in npz json; do
        if remote_ssh "[ -f '$REMOTE_BASE/results/$group/$name.$extension' ]" 2>/dev/null; then
            copy_relative_path "results/$group/$name.$extension"
            copied=1
        fi
    done
    if [ "$copied" -eq 0 ]; then
        echo "Missing remote result bundle: results/$group/$name.{npz,json}" >&2
        return 1
    fi
}

copy_named_result() {
    local group="$1"
    local name="$2"
    validate_component "$group" "result group"
    validate_component "$name" "result name"
    if remote_ssh "[ -e '$REMOTE_BASE/results/$group/$name' ]" 2>/dev/null; then
        copy_relative_path "results/$group/$name"
    else
        copy_bundle "$group" "$name"
    fi
}

list_groups() {
    echo "Remote result groups at $REMOTE_TARGET:$REMOTE_BASE/results"
    remote_ssh \
        "if [ ! -d '$REMOTE_BASE/results' ]; then echo 'Missing remote results directory' >&2; exit 4; fi; find '$REMOTE_BASE/results' -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort"
}

list_group() {
    local group="$1"
    local limit="${2:-20}"
    local rows shown=0
    validate_component "$group" "result group"
    is_positive_integer "$limit" || fail "list count must be a positive integer"
    echo "Newest entries in $REMOTE_TARGET:$REMOTE_BASE/results/$group"
    rows="$(remote_rows "$group")"
    while IFS=$'\t' read -r mtime kind name; do
        [ -n "$name" ] || continue
        printf '  %s  %s\n' "$kind" "$name"
        shown=$((shown + 1))
        [ "$shown" -ge "$limit" ] && break
    done <<< "$rows"
}

copy_latest() {
    local group="$1"
    local count="$2"
    local row_mtime row_kind row_name key existing_index index member rows
    local candidate_count=0
    local candidate_keys=()
    local candidate_types=()
    local candidate_members=()

    validate_component "$group" "result group"
    is_positive_integer "$count" || fail "latest count must be a positive integer"

    rows="$(remote_rows "$group")"
    while IFS=$'\t' read -r row_mtime row_kind row_name; do
        [ -n "$row_name" ] || continue
        if ! [[ "$row_name" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
            echo "Skipping unsupported remote entry name: $row_name" >&2
            continue
        fi
        if [ "$row_kind" = "d" ]; then
            key="D:$row_name"
        elif [ "$row_kind" = "f" ] && [[ "$row_name" =~ ^(.+)\.(npz|json)$ ]]; then
            key="P:${BASH_REMATCH[1]}"
        elif [ "$row_kind" = "f" ]; then
            key="F:$row_name"
        else
            continue
        fi

        existing_index=-1
        for ((index = 0; index < candidate_count; index++)); do
            if [ "${candidate_keys[$index]}" = "$key" ]; then
                existing_index=$index
                break
            fi
        done
        if [ "$existing_index" -ge 0 ]; then
            candidate_members[$existing_index]="${candidate_members[$existing_index]} $row_name"
            continue
        fi
        candidate_keys[$candidate_count]="$key"
        candidate_types[$candidate_count]="${key%%:*}"
        candidate_members[$candidate_count]="$row_name"
        candidate_count=$((candidate_count + 1))
    done <<< "$rows"

    if [ "$candidate_count" -eq 0 ]; then
        fail "no results found in remote group '$group'"
    fi
    if [ "$count" -gt "$candidate_count" ]; then
        count="$candidate_count"
    fi

    echo "Copying latest $count result(s) from '$group':"
    for ((index = 0; index < count; index++)); do
        echo "  - ${candidate_keys[$index]#*:}"
    done
    for ((index = 0; index < count; index++)); do
        case "${candidate_types[$index]}" in
            D|F)
                copy_relative_path "results/$group/${candidate_members[$index]}"
                ;;
            P)
                for member in ${candidate_members[$index]}; do
                    copy_relative_path "results/$group/$member"
                done
                ;;
        esac
    done
}

echo "Remote: $REMOTE_TARGET:$REMOTE_BASE"
if [ "$DRY_RUN" = true ]; then
    echo "Mode:   dry-run (no files will be copied)"
fi

case "$1" in
    --groups)
        [ "$#" -eq 1 ] || fail "--groups takes no arguments"
        list_groups
        ;;
    --list)
        if [ "$#" -eq 1 ]; then
            list_groups
        elif [ "$#" -eq 2 ] || [ "$#" -eq 3 ]; then
            group="$(resolve_group "$2")"
            list_group "$group" "${3:-20}"
        else
            fail "usage: --list [group [N]]"
        fi
        ;;
    --latest)
        group="wr2_walking"
        count=1
        if [ "$#" -ge 2 ]; then
            if is_positive_integer "$2"; then
                [ "$#" -eq 2 ] || fail "usage: --latest [group] [N]"
                count="$2"
            else
                group="$(resolve_group "$2")"
                count="${3:-1}"
            fi
        fi
        [ "$#" -le 3 ] || fail "usage: --latest [group] [N]"
        copy_latest "$group" "$count"
        ;;
    --copy)
        [ "$#" -eq 3 ] || fail "usage: --copy <group> <name>"
        group="$(resolve_group "$2")"
        copy_named_result "$group" "$3"
        ;;
    --*)
        fail "unknown option '$1'"
        ;;
    *)
        [ "$#" -eq 1 ] || fail "a repository-relative path must be one argument"
        copy_relative_path "$1"
        ;;
esac
